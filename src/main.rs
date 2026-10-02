#![no_std]
#![no_main]

pub mod bitpack;
pub mod boot;
pub mod io_ring;
pub mod kernel;
pub mod nvme;
pub mod pci;
pub mod serial;
pub mod shared_mem;
pub mod timer;

use core::fmt::Write;
use core::panic::PanicInfo;
use uefi::prelude::*;

use crate::bitpack::{
    parse_neur_header, validate_ternary_payload, NeurHeader, NeurHeaderError,
    NEUR_ATTENTION_PAYLOAD_BYTES, NEUR_HEADER_SIZE, NEUR_MAGIC, NEUR_MLP_PAYLOAD_BYTES,
    NEUR_MODEL_CAUSAL_LINEAR_ATTENTION, NEUR_MODEL_MLP,
};
use crate::boot::{establish_cpu_sovereignty, exit_uefi_boot_services, hardware_shutdown};
use crate::io_ring::{InputFrame, InputRing, OutputFrame, OutputRing};
use crate::kernel::{
    causal_linear_attention_scalar, enable_avx512_os_state, reset_attention_state,
    ternary_mlp_avx512, ternary_mlp_scalar, AlignedInputs64, TernaryLinearAttention,
    TernaryMlp64x32x16, LINEAR_ATTENTION_STATE_LEN, MLP_OUTPUT_DIM,
};
use crate::timer::{
    cycles_for_seconds, cycles_to_micros_milli, read_tsc, summarize, LatencySample, LatencySummary,
    MAX_LATENCY_SAMPLES,
};

const STREAM_FRAME_LIMIT: u64 = 8;
const STREAM_TIMEOUT_SECONDS: u64 = 25;

/// 4 KiB page-aligned DMA buffer, large enough for 512-byte and 4 KiB namespace blocks.
#[repr(C, align(4096))]
pub struct WeightBuffer(pub [u8; nvme::DMA_PAGE_SIZE]);

/// 64-byte aligned universal streaming I/O ring buffers.
static mut INPUT_RING: InputRing = InputRing::new();
static mut OUTPUT_RING: OutputRing = OutputRing::new();

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("[PANIC]: {}", info);
    loop {
        core::hint::spin_loop();
    }
}

#[derive(Clone, Copy)]
enum LoadedModel {
    Mlp(TernaryMlp64x32x16),
    Attention(TernaryLinearAttention),
}

fn infer_values(
    inputs: &[i8; 64],
    model: &LoadedModel,
    attention_state: &mut [i32; LINEAR_ATTENTION_STATE_LEN],
    avx512_enabled: bool,
    outputs: &mut [i32; 64],
) -> u8 {
    match model {
        LoadedModel::Mlp(mlp) => {
            let aligned_inputs = AlignedInputs64(*inputs);
            let results = if avx512_enabled {
                unsafe { ternary_mlp_avx512(&aligned_inputs, mlp) }
            } else {
                ternary_mlp_scalar(inputs, mlp)
            };
            outputs[..MLP_OUTPUT_DIM].copy_from_slice(&results);
            MLP_OUTPUT_DIM as u8
        }
        LoadedModel::Attention(attention) => {
            let results = causal_linear_attention_scalar(inputs, attention, attention_state);
            outputs[..results.len()].copy_from_slice(&results);
            results.len() as u8
        }
    }
}

fn infer_frame(
    frame: InputFrame,
    model: &LoadedModel,
    attention_state: &mut [i32; LINEAR_ATTENTION_STATE_LEN],
    avx512_enabled: bool,
) -> OutputFrame {
    let mut output = OutputFrame {
        output_dim: 0,
        values: [0; 64],
        t0_preamble: frame.t0_preamble,
        t1_ingress: frame.t1_ingress,
        t2_compute: 0,
    };
    output.output_dim = infer_values(
        &frame.payload,
        model,
        attention_state,
        avx512_enabled,
        &mut output.values,
    );
    output
}

unsafe fn transmit_output_frame(frame: &OutputFrame) -> u64 {
    serial::write_raw_bytes_from(serial::COM2_BASE, &serial::OUTPUT_PREAMBLE);
    serial::write_byte_raw_from(serial::COM2_BASE, frame.output_dim);
    let mut index = 0usize;
    while index < frame.output_dim as usize {
        serial::write_raw_bytes_from(serial::COM2_BASE, &frame.values[index].to_le_bytes());
        index += 1;
    }
    read_tsc()
}

fn summarize_stage(samples: &[LatencySample], stage: u8) -> LatencySummary {
    let mut values = [0u64; MAX_LATENCY_SAMPLES];
    let count = samples.len().min(MAX_LATENCY_SAMPLES);
    let mut index = 0usize;
    while index < count {
        values[index] = match stage {
            0 => samples[index].ingress_cycles,
            1 => samples[index].compute_cycles,
            2 => samples[index].egress_cycles,
            _ => samples[index].turnaround_cycles,
        };
        index += 1;
    }
    summarize(&values[..count])
}

fn log_latency_summary(name: &str, summary: LatencySummary) {
    let min_us = cycles_to_micros_milli(summary.min);
    let p50_us = cycles_to_micros_milli(summary.p50);
    let p99_us = cycles_to_micros_milli(summary.p99);
    let max_us = cycles_to_micros_milli(summary.max);
    serial_println!(
        "[LATENCY]: {} cycles min={} p50={} p99={} max={}; calibrated_us min={}.{:03} p50={}.{:03} p99={}.{:03} max={}.{:03}",
        name,
        summary.min,
        summary.p50,
        summary.p99,
        summary.max,
        min_us / 1000,
        min_us % 1000,
        p50_us / 1000,
        p50_us % 1000,
        p99_us / 1000,
        p99_us % 1000,
        max_us / 1000,
        max_us % 1000
    );
}

fn verify_shared_mailbox(
    mailbox: &shared_mem::SharedMailbox,
    model: &TernaryMlp64x32x16,
    avx512_enabled: bool,
) -> bool {
    const TEST_FRAMES: usize = shared_mem::MAILBOX_CAPACITY;
    let mut samples = [LatencySample::default(); TEST_FRAMES];
    let mut completed = 0usize;
    let mut dropped = 0usize;

    while completed < TEST_FRAMES {
        let test_payload = [1i8; 64];
        let t0_publish = unsafe { read_tsc() };
        if mailbox
            .try_publish_input(&test_payload, t0_publish)
            .is_err()
        {
            dropped += 1;
            continue;
        }

        let (sequence, input_slot, t0_observed) = loop {
            if let Some(pending) = mailbox.peek_input() {
                break pending;
            }
            core::arch::x86_64::_mm_pause();
        };
        let t1_ingest = unsafe { read_tsc() };
        let published = mailbox.compute_and_publish_output(
            MLP_OUTPUT_DIM,
            t0_observed,
            t1_ingest,
            |output_values| {
                let input = unsafe { input_slot.payload() };
                let aligned_inputs = AlignedInputs64(*input);
                let results = if avx512_enabled {
                    unsafe { ternary_mlp_avx512(&aligned_inputs, model) }
                } else {
                    ternary_mlp_scalar(input, model)
                };
                let mut row = 0usize;
                while row < MLP_OUTPUT_DIM {
                    output_values[row] = results[row];
                    row += 1;
                }
                unsafe { read_tsc() }
            },
        );
        mailbox.consume_input(sequence);
        let (_, t2_compute, t3_commit) = match published {
            Ok(timestamps) => timestamps,
            Err(()) => {
                dropped += 1;
                continue;
            }
        };

        let (_, output_values, actual_dim) = match mailbox.consume_output() {
            Some(output) => output,
            None => {
                dropped += 1;
                continue;
            }
        };
        if actual_dim as usize != MLP_OUTPUT_DIM {
            return false;
        }

        let expected = ternary_mlp_scalar(&test_payload, model);
        let mut row = 0usize;
        while row < MLP_OUTPUT_DIM {
            if output_values[row] != expected[row] {
                return false;
            }
            row += 1;
        }

        samples[completed] = LatencySample {
            ingress_cycles: t1_ingest.saturating_sub(t0_observed),
            compute_cycles: t2_compute.saturating_sub(t1_ingest),
            egress_cycles: t3_commit.saturating_sub(t2_compute),
            turnaround_cycles: t3_commit.saturating_sub(t0_observed),
        };
        completed += 1;
    }

    if dropped != 0 {
        serial_println!("[SHM TEST ERROR]: dropped={} frames", dropped);
        return false;
    }
    log_latency_summary("shared_mailbox_ingress", summarize_stage(&samples, 0));
    log_latency_summary("shared_mailbox_compute", summarize_stage(&samples, 1));
    log_latency_summary("shared_mailbox_commit", summarize_stage(&samples, 2));
    let summary = summarize_stage(&samples, 3);
    log_latency_summary("shared_mailbox_turnaround", summary);
    let sub_microsecond =
        timer::tsc_frequency_hz() != 0 && summary.max < (timer::tsc_frequency_hz() / 1_000_000);
    serial_println!(
        "[MLP VERIFY]: Layer1=64->32, hard_sign=32, Layer2=32->16; frames={}, drops={}, scalar_reference=match, max_below_1us={}",
        completed,
        dropped,
        sub_microsecond
    );
    true
}

#[derive(Clone, Copy, Debug)]
enum ShardLoadError {
    NotFound,
    InvalidHeader(NeurHeaderError),
    InvalidPayload,
    ReadFailed,
}

fn parse_neur_shard(data: &[u8]) -> Result<(LoadedModel, NeurHeader), ShardLoadError> {
    let header = parse_neur_header(data).map_err(ShardLoadError::InvalidHeader)?;
    let payload_size = match header.model_type {
        NEUR_MODEL_MLP => NEUR_MLP_PAYLOAD_BYTES,
        NEUR_MODEL_CAUSAL_LINEAR_ATTENTION => NEUR_ATTENTION_PAYLOAD_BYTES,
        _ => return Err(ShardLoadError::InvalidPayload),
    };
    let payload_end = NEUR_HEADER_SIZE + payload_size;
    if payload_end > data.len() || payload_end > nvme::DMA_PAGE_SIZE {
        return Err(ShardLoadError::InvalidPayload);
    }
    let payload = &data[NEUR_HEADER_SIZE..payload_end];
    if !validate_ternary_payload(payload) {
        return Err(ShardLoadError::InvalidPayload);
    }

    let model = match header.model_type {
        NEUR_MODEL_MLP => LoadedModel::Mlp(
            TernaryMlp64x32x16::from_packed_payload(payload)
                .ok_or(ShardLoadError::InvalidPayload)?,
        ),
        NEUR_MODEL_CAUSAL_LINEAR_ATTENTION => LoadedModel::Attention(
            TernaryLinearAttention::from_packed_payload(payload)
                .ok_or(ShardLoadError::InvalidPayload)?,
        ),
        _ => return Err(ShardLoadError::InvalidPayload),
    };
    Ok((model, header))
}

unsafe fn load_neur_shard(
    controller: &mut nvme::NvmeController,
    buffer: &mut WeightBuffer,
) -> Result<(LoadedModel, NeurHeader, u64), ShardLoadError> {
    let logical_block_size = controller.logical_block_size;
    if !(NEUR_HEADER_SIZE..=nvme::DMA_PAGE_SIZE).contains(&logical_block_size) {
        return Err(ShardLoadError::InvalidPayload);
    }
    let candidate_byte_offsets = [133120u64 * 512, 34816u64 * 512, 67584u64 * 512, 0];
    let mut selected_lba = None;

    for byte_offset in candidate_byte_offsets {
        if byte_offset % logical_block_size as u64 != 0 {
            continue;
        }
        let lba = byte_offset / logical_block_size as u64;
        buffer.0.fill(0);
        if controller
            .read_raw_lba(lba, 1, buffer.0.as_mut_ptr(), buffer.0.len())
            .is_err()
        {
            continue;
        }

        let magic = u32::from_be_bytes([buffer.0[0], buffer.0[1], buffer.0[2], buffer.0[3]]);
        if magic == NEUR_MAGIC {
            selected_lba = Some(lba);
            break;
        }
    }

    let lba = selected_lba.ok_or(ShardLoadError::NotFound)?;
    let header = parse_neur_header(&buffer.0[..logical_block_size])
        .map_err(ShardLoadError::InvalidHeader)?;
    let payload_size = match header.model_type {
        NEUR_MODEL_MLP => NEUR_MLP_PAYLOAD_BYTES,
        NEUR_MODEL_CAUSAL_LINEAR_ATTENTION => NEUR_ATTENTION_PAYLOAD_BYTES,
        _ => return Err(ShardLoadError::InvalidPayload),
    };
    let payload_end = NEUR_HEADER_SIZE + payload_size;
    if payload_end > nvme::DMA_PAGE_SIZE {
        return Err(ShardLoadError::InvalidPayload);
    }
    let block_count = (payload_end + logical_block_size - 1) / logical_block_size;
    if block_count == 0 || block_count > u16::MAX as usize {
        return Err(ShardLoadError::InvalidPayload);
    }

    buffer.0.fill(0);
    controller
        .read_raw_lba(
            lba,
            block_count as u16,
            buffer.0.as_mut_ptr(),
            buffer.0.len(),
        )
        .map_err(|_| ShardLoadError::ReadFailed)?;

    let (model, confirmed_header) = parse_neur_shard(&buffer.0[..payload_end])?;
    if confirmed_header != header {
        return Err(ShardLoadError::InvalidPayload);
    }
    Ok((model, header, lba))
}

unsafe fn load_neur_shard_from_nvme(
    buffer: &mut WeightBuffer,
) -> Result<(LoadedModel, NeurHeader, u64, usize), ShardLoadError> {
    let pci_nvme = pci::find_nvme_device().ok_or(ShardLoadError::NotFound)?;
    serial_println!(
        "[SOVEREIGN_CORE]: Found NVMe on PCI {:02x}:{:02x}.{:x} BAR0: {:#018x}",
        pci_nvme.bus,
        pci_nvme.slot,
        pci_nvme.func,
        pci_nvme.bar0_phys
    );
    let mut controller =
        nvme::NvmeController::init(&pci_nvme).map_err(|_| ShardLoadError::ReadFailed)?;
    let block_size = controller.logical_block_size;
    let (model, header, lba) = load_neur_shard(&mut controller, buffer)?;
    Ok((model, header, lba, block_size))
}

fn safe_identity_model() -> LoadedModel {
    let mut model = TernaryMlp64x32x16::zeroed();
    let mut index = 0usize;
    while index < MLP_OUTPUT_DIM {
        let byte = index >> 2;
        let shift = (index & 3) * 2;
        model.layer1[index][byte] |= 0b01 << shift;
        model.layer2[index][byte] |= 0b01 << shift;
        index += 1;
    }
    LoadedModel::Mlp(model)
}

fn run_host_ipc_loop(
    mailbox: &shared_mem::SharedMailbox,
    model: &LoadedModel,
    avx512_enabled: bool,
) -> bool {
    let start = unsafe { read_tsc() };
    let timeout_cycles = cycles_for_seconds(STREAM_TIMEOUT_SECONDS);
    let mut completed = 0u64;
    let mut drops = 0u64;
    let mut samples = [LatencySample::default(); MAX_LATENCY_SAMPLES];
    let mut cursor = 0usize;
    let mut attention_state = [0i32; LINEAR_ATTENTION_STATE_LEN];

    serial_println!(
        "[SHM HOST IPC]: Waiting for host frames at GPA {:#018x}; limit={}, timeout={} s.",
        shared_mem::HOST_MAILBOX_PHYSICAL_BASE,
        STREAM_FRAME_LIMIT,
        STREAM_TIMEOUT_SECONDS
    );

    while completed < STREAM_FRAME_LIMIT
        && unsafe { read_tsc() }.saturating_sub(start) < timeout_cycles
    {
        if let Some((sequence, input_slot, t0_ready)) = mailbox.peek_input() {
            let t1_ingest = unsafe { read_tsc() };
            let published = mailbox.compute_and_publish_output(
                match model {
                    LoadedModel::Mlp(_) => MLP_OUTPUT_DIM,
                    LoadedModel::Attention(_) => LINEAR_ATTENTION_STATE_LEN / 16,
                },
                t0_ready,
                t1_ingest,
                |output_values| {
                    let inputs = unsafe { input_slot.payload() };
                    infer_values(
                        inputs,
                        model,
                        &mut attention_state,
                        avx512_enabled,
                        output_values,
                    );
                    unsafe { read_tsc() }
                },
            );
            mailbox.consume_input(sequence);
            match published {
                Ok((_, t2_compute, t3_commit)) => {
                    samples[cursor] = LatencySample {
                        ingress_cycles: t1_ingest.saturating_sub(t0_ready),
                        compute_cycles: t2_compute.saturating_sub(t1_ingest),
                        egress_cycles: t3_commit.saturating_sub(t2_compute),
                        turnaround_cycles: t3_commit.saturating_sub(t0_ready),
                    };
                    cursor = (cursor + 1) % MAX_LATENCY_SAMPLES;
                    completed += 1;
                }
                Err(()) => drops += 1,
            }
        } else {
            core::arch::x86_64::_mm_pause();
        }
    }

    if cursor > 0 {
        let count = (completed as usize).min(MAX_LATENCY_SAMPLES);
        let window = &samples[..count];
        log_latency_summary("host_ipc_ingress", summarize_stage(window, 0));
        log_latency_summary("host_ipc_compute", summarize_stage(window, 1));
        log_latency_summary("host_ipc_commit", summarize_stage(window, 2));
        log_latency_summary("host_ipc_turnaround", summarize_stage(window, 3));
    }
    serial_println!(
        "[SHM HOST IPC]: guest_processed={}, drops={}, elapsed_cycles={}",
        completed,
        drops,
        unsafe { read_tsc() }.saturating_sub(start)
    );
    completed == STREAM_FRAME_LIMIT && drops == 0
}

#[entry]
fn main(
    image_handle: uefi::Handle,
    mut system_table: uefi::table::SystemTable<uefi::table::Boot>,
) -> uefi::Status {
    // 1. Direct Hardware Serial Telemetry Initialization (COM1 115200 8-N-1)
    unsafe {
        serial::init();
        serial::init_wire_port();
    }
    serial_println!("[SOVEREIGN_CORE]: SERIAL TELEMETRY INITIALIZED (COM1 115200 8-N-1)");

    let calibrated_hz = timer::calibrate_tsc(|window_us| {
        let _ = system_table.boot_services().stall(window_us);
    });
    let (frequency_mhz, frequency_hundredths) = timer::tsc_frequency_mhz_parts();
    if timer::calibration_used_cpuid15() {
        serial_println!(
            "[TIMER]: Calibrated TSC frequency: {}.{:02} MHz (CPUID leaf 0x15)",
            frequency_mhz,
            frequency_hundredths
        );
    } else {
        serial_println!(
            "[TIMER]: Calibrated TSC frequency: {}.{:02} MHz (100 ms UEFI stall)",
            frequency_mhz,
            frequency_hundredths
        );
    }
    let (invariant_tsc, rdtscp_supported) = boot::tsc_capabilities();
    serial_println!(
        "[TIMER]: Invariant TSC={}, RDTSCP={}, reads serialized with LFENCE.",
        invariant_tsc,
        rdtscp_supported
    );
    if !invariant_tsc {
        serial_println!("[TIMER WARNING]: TSC is not invariant; calibrated microseconds may drift if CPU power state changes.");
    }
    let (hwp_supported, hwp_enabled, turbo_supported) = boot::cpu_power_management_status();
    serial_println!(
        "[CPU POWER]: HWP supported={}, enabled={}, turbo capability={}; firmware/thermal limits left unchanged.",
        hwp_supported,
        hwp_enabled,
        turbo_supported
    );
    if calibrated_hz == 0 {
        serial_println!("[TIMER WARNING]: TSC frequency calibration failed; calibrated time conversions unavailable.");
    }

    let (guest_f, guest_bw, guest_dq, guest_vl, guest_xsave, guest_osxsave, guest_xcr0) =
        boot::avx512_guest_features();
    serial_println!(
        "[GUEST CPUID]: AVX512F={}, AVX512BW={}, AVX512DQ={}, AVX512VL={}, XSAVE={}, OSXSAVE={}, XCR0=0x{:016X}",
        guest_f,
        guest_bw,
        guest_dq,
        guest_vl,
        guest_xsave,
        guest_osxsave,
        guest_xcr0
    );
    let avx512_enabled = if boot::cpu_supports_avx512() {
        let state_enabled = unsafe { enable_avx512_os_state() };
        if state_enabled {
            serial_println!("[CPU]: AVX512F/BW/VL/DQ and OS ZMM state available.");
        } else {
            serial_println!(
                "[CPU WARNING]: AVX-512 state unavailable; using scalar compute fallback."
            );
        }
        state_enabled
    } else {
        serial_println!("[CPU WARNING]: AVX-512 unavailable; using scalar compute fallback.");
        false
    };
    serial_println!(
        "[SIMD]: MLP backend selected={}",
        if avx512_enabled { "AVX-512" } else { "scalar" }
    );

    let mailbox_ptr = match boot::reserve_shared_mailbox() {
        Ok(pointer) => pointer,
        Err(error) => {
            serial_println!("[SHM ERROR]: UEFI page allocation failed: {:?}", error);
            return uefi::Status::OUT_OF_RESOURCES;
        }
    };
    let mailbox_address = mailbox_ptr.as_ptr() as u64;
    serial_println!(
        "[SHM]: Initialized Mailbox at physical addr {:#018x}, bytes={}, pages={}, alignment={}, mode={}",
        mailbox_address,
        core::mem::size_of::<shared_mem::SharedMailbox>(),
        core::mem::size_of::<shared_mem::SharedMailbox>().div_ceil(4096),
        core::mem::align_of::<shared_mem::SharedMailbox>(),
        if cfg!(feature = "host-ipc") { "host-backed GPA" } else { "UEFI allocated" }
    );

    let stdout = system_table.stdout();
    let _ = stdout.reset(false);
    let _ = writeln!(stdout, "NEURAL-BOX CORE: HARNESS INITIALIZED");

    // UEFI filesystem protocols are unavailable after ExitBootServices. Keep
    // the DMA-aligned shard buffer alive and fill it before transferring control.
    let mut weight_buffer = WeightBuffer([0u8; nvme::DMA_PAGE_SIZE]);
    let filesystem_shard_len = boot::read_weights_file(&mut weight_buffer.0);
    match filesystem_shard_len {
        Some(bytes) => serial_println!(
            "[LOADER]: Read weights.bin from UEFI SimpleFileSystem ({} bytes).",
            bytes
        ),
        None => serial_println!("[LOADER]: File weights.bin not found; using fallback"),
    }

    // 3. Exit UEFI Boot Services & Ingest Memory Map
    let mmap_info = unsafe { exit_uefi_boot_services(image_handle, &mut system_table) };

    // 4. Absolute CPU Sovereignty Configuration (Mute interrupts, verify CR0/CR4/XCR0)
    unsafe {
        establish_cpu_sovereignty(avx512_enabled);
    }
    serial_println!("[TIMER]: Maskable interrupts confirmed disabled after ExitBootServices.");
    serial_println!("[SOVEREIGN_CORE]: BOOT SERVICES TERMINATED. INTERRUPTS MUTED. CPU ACQUIRED.");
    serial_println!(
        "[SOVEREIGN_CORE]: PHYSICAL MMAP INGESTED ({} entries, {} bytes)",
        mmap_info.entry_count,
        mmap_info.map_size
    );

    let load_result = if let Some(file_bytes) = filesystem_shard_len {
        match parse_neur_shard(&weight_buffer.0[..file_bytes]) {
            Ok((loaded_model, header)) => {
                serial_println!(
                    "[LOADER]: Validated file shard from UEFI FAT volume ({} bytes).",
                    file_bytes
                );
                Ok((loaded_model, header, None))
            }
            Err(error) => {
                serial_println!(
                    "[LOADER]: weights.bin is invalid ({:?}); trying raw NVMe fallback.",
                    error
                );
                unsafe { load_neur_shard_from_nvme(&mut weight_buffer) }.map(
                    |(model, header, lba, block_size)| (model, header, Some((lba, block_size))),
                )
            }
        }
    } else {
        unsafe { load_neur_shard_from_nvme(&mut weight_buffer) }
            .map(|(model, header, lba, block_size)| (model, header, Some((lba, block_size))))
    };
    let model = match load_result {
        Ok((loaded_model, header, source)) => {
            serial_println!(
                "[SHARD]: MAGIC=0x{:08X}, VERSION={}, INPUT_DIM={}, MODEL_TYPE={}, HIDDEN_OR_ATTN_DIM={}, OUTPUT_DIM={}",
                NEUR_MAGIC,
                header.version,
                header.input_dim,
                header.model_type,
                header.hidden_dim,
                header.output_dim
            );
            if let Some((lba, block_size)) = source {
                serial_println!(
                    "[LOADER]: Loaded shard from raw NVMe LBA {} (block size={} bytes).",
                    lba,
                    block_size
                );
            } else {
                serial_println!("[LOADER]: Using model shard from UEFI FAT filesystem.");
            }
            match loaded_model {
                LoadedModel::Mlp(_) => {
                    serial_println!("[SHARD]: Model=MLP; Layer 1: 64 -> 32, packed_weights=512 bytes.");
                    serial_println!("[SHARD]: Layer 2: 32 -> 16, packed_weights=128 bytes.");
                }
                LoadedModel::Attention(_) => serial_println!(
                    "[SHARD]: Model=causal-linear-attention; Q/K/V=64 -> 16, O=16 -> 16, packed_weights={} bytes.",
                    NEUR_ATTENTION_PAYLOAD_BYTES
                ),
            }
            loaded_model
        }
        Err(error) => {
            serial_println!(
                "[LOADER]: NVMe fallback unavailable or invalid ({:?}); using safe identity model.",
                error
            );
            match error {
                ShardLoadError::InvalidHeader(header_error) => serial_println!(
                    "[SHARD ERROR]: Invalid model header ({:?}); using safe identity model.",
                    header_error
                ),
                ShardLoadError::NotFound => {
                    serial_println!("[SHARD ERROR]: No valid NEUR shard; using safe identity model.")
                }
                ShardLoadError::InvalidPayload => serial_println!(
                    "[SHARD ERROR]: Model payload bounds/encoding invalid; using safe identity model."
                ),
                ShardLoadError::ReadFailed => {
                    serial_println!("[SHARD ERROR]: Model NVMe read failed; using safe identity model.")
                }
            }
            safe_identity_model()
        }
    };
    serial_println!("[SOVEREIGN_CORE]: DMA SHARD INGESTION COMPLETE.");
    serial_println!(
        "[ALLOC]: Model inference uses fixed-size static/stack buffers; no heap allocator."
    );

    let mailbox = unsafe { mailbox_ptr.as_ref() };
    if !cfg!(feature = "host-ipc") {
        if mailbox.magic != shared_mem::MAILBOX_MAGIC
            || mailbox.version != shared_mem::MAILBOX_VERSION
            || (matches!(model, LoadedModel::Mlp(_))
                && !verify_shared_mailbox(
                    mailbox,
                    match &model {
                        LoadedModel::Mlp(mlp) => mlp,
                        LoadedModel::Attention(_) => unreachable!(),
                    },
                    avx512_enabled,
                ))
        {
            serial_println!("[SHM TEST ERROR]: mailbox validation or zero-copy inference failed.");
            return uefi::Status::ABORTED;
        }
    }

    if cfg!(feature = "host-ipc") {
        if mailbox.magic != shared_mem::MAILBOX_MAGIC
            || mailbox.version != shared_mem::MAILBOX_VERSION
            || !run_host_ipc_loop(mailbox, &model, avx512_enabled)
        {
            serial_println!("[SHM HOST IPC ERROR]: mailbox validation, host frame count, or output commit failed.");
            return uefi::Status::ABORTED;
        }
        unsafe { hardware_shutdown() };
    }

    let calibrated_timeout_cycles = cycles_for_seconds(STREAM_TIMEOUT_SECONDS);
    let stream_timeout_cycles = if calibrated_timeout_cycles == 0 {
        u64::MAX
    } else {
        calibrated_timeout_cycles
    };
    serial_println!(
        "[UART]: COM2 protocol RX=NB+64 bytes or standalone NR reset, TX=NR+dim+little-endian i32; frame_limit={}, timeout={} s ({} cycles)",
        STREAM_FRAME_LIMIT,
        STREAM_TIMEOUT_SECONDS,
        stream_timeout_cycles
    );
    serial_println!("[RING]: Atomic SPSC input/output queues ready (capacity=128).");

    let (mut input_producer, mut input_consumer) =
        unsafe { (&mut *core::ptr::addr_of_mut!(INPUT_RING)).split() };
    let (mut output_producer, mut output_consumer) =
        unsafe { (&mut *core::ptr::addr_of_mut!(OUTPUT_RING)).split() };
    unsafe {
        serial::enable_rx_interrupt(serial::COM2_BASE);
        serial::write_raw_bytes_from(serial::COM2_BASE, b"UART_READY\n");
    }

    let mut frame_reader = serial::FrameReader::new();
    let mut attention_state = [0i32; LINEAR_ATTENTION_STATE_LEN];
    let stream_start = unsafe { read_tsc() };
    let mut stream_events = 0u64;
    let mut reset_commands = 0u64;
    let mut reset_seen = false;
    let mut frames_ingress = 0u64;
    let mut frames_transmitted = 0u64;
    let mut ring_full_drops = 0u64;
    let mut latency_samples = [LatencySample::default(); MAX_LATENCY_SAMPLES];
    let mut latency_sample_count = 0usize;
    let mut latency_sample_cursor = 0usize;
    let mut latency_samples_total = 0u64;

    while !reset_seen
        && (STREAM_FRAME_LIMIT == 0 || stream_events < STREAM_FRAME_LIMIT)
        && unsafe { read_tsc() }.saturating_sub(stream_start) < stream_timeout_cycles
    {
        if let Some(event) = frame_reader.poll_frame(serial::COM2_BASE) {
            match event {
                serial::FrameEvent::Input(received) => {
                    let frame = InputFrame {
                        payload: received.payload,
                        t0_preamble: received.preamble_tsc,
                        t1_ingress: 0,
                    };
                    if input_producer
                        .push_with(frame, |queued| queued.t1_ingress = unsafe { read_tsc() })
                        .is_ok()
                    {
                        frames_ingress += 1;
                        stream_events += 1;
                    } else {
                        ring_full_drops += 1;
                    }
                }
                serial::FrameEvent::Reset { .. } => {
                    reset_attention_state(&mut attention_state);
                    reset_commands += 1;
                    stream_events += 1;
                    reset_seen = true;
                    serial_println!(
                        "[ATTENTION RESET]: command=NR state00={} reset_count={}",
                        attention_state[0],
                        reset_commands
                    );
                }
            }
        }

        if let Some(frame) = input_consumer.pop() {
            let result = infer_frame(frame, &model, &mut attention_state, avx512_enabled);
            if matches!(model, LoadedModel::Attention(_)) {
                serial_println!(
                    "[ATTENTION STATE]: frame={} state00={}",
                    frames_transmitted + 1,
                    attention_state[0]
                );
            }

            if output_producer
                .push_with(result, |queued| queued.t2_compute = unsafe { read_tsc() })
                .is_err()
            {
                ring_full_drops += 1;
            } else if let Some(output) = output_consumer.pop() {
                let t3_uart_tx = unsafe { transmit_output_frame(&output) };
                latency_samples[latency_sample_cursor] = LatencySample {
                    ingress_cycles: output.t1_ingress.saturating_sub(output.t0_preamble),
                    compute_cycles: output.t2_compute.saturating_sub(output.t1_ingress),
                    egress_cycles: t3_uart_tx.saturating_sub(output.t2_compute),
                    turnaround_cycles: t3_uart_tx.saturating_sub(output.t0_preamble),
                };
                latency_sample_cursor = (latency_sample_cursor + 1) % MAX_LATENCY_SAMPLES;
                latency_sample_count = (latency_sample_count + 1).min(MAX_LATENCY_SAMPLES);
                latency_samples_total = latency_samples_total.wrapping_add(1);
                frames_transmitted += 1;
            }
        } else {
            core::hint::spin_loop();
        }
    }

    let elapsed_cycles = unsafe { read_tsc() }.saturating_sub(stream_start).max(1);
    let throughput_milli_fps = if calibrated_hz == 0 {
        0
    } else {
        ((frames_transmitted as u128 * calibrated_hz as u128 * 1000) / elapsed_cycles as u128)
            .min(u64::MAX as u128) as u64
    };
    serial_println!(
        "[UART]: RX frames={}, TX frames={}, reset_commands={}, stream_events={}, ring_full_drops={}, elapsed_cycles={}, throughput={}.{:03} frames/s",
        frames_ingress,
        frames_transmitted,
        reset_commands,
        stream_events,
        ring_full_drops,
        elapsed_cycles,
        throughput_milli_fps / 1000,
        throughput_milli_fps % 1000
    );
    if latency_sample_count > 0 {
        let samples = &latency_samples[..latency_sample_count];
        serial_println!(
            "[LATENCY]: rolling samples={} of total processed frames={}",
            latency_sample_count,
            latency_samples_total
        );
        log_latency_summary("ingress", summarize_stage(samples, 0));
        log_latency_summary("compute", summarize_stage(samples, 1));
        log_latency_summary("egress", summarize_stage(samples, 2));
        log_latency_summary("turnaround", summarize_stage(samples, 3));
    } else {
        serial_println!("[LATENCY]: No completed frames; no latency samples available.");
    }
    serial_println!(
        "[RING]: Atomic SPSC ingress/egress operations completed without lock or panic."
    );

    // 9. Clean Hardware Exit
    unsafe {
        hardware_shutdown();
    }
}
