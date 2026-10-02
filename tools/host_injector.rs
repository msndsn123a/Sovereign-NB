#![cfg_attr(not(windows), allow(dead_code))]

#[cfg(not(windows))]
compile_error!("host_injector is a Windows-only shared mailbox producer");

#[cfg(windows)]
mod windows_host {
    use std::ffi::c_void;
    use std::fs;
    use std::mem;
    use std::ptr;
    use std::sync::atomic::{fence, AtomicU32, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    type Handle = *mut c_void;

    const GENERIC_READ: u32 = 0x8000_0000;
    const GENERIC_WRITE: u32 = 0x4000_0000;
    const FILE_SHARE_READ: u32 = 1;
    const FILE_SHARE_WRITE: u32 = 2;
    const OPEN_EXISTING: u32 = 3;
    const PAGE_READWRITE: u32 = 0x04;
    const FILE_MAP_ALL_ACCESS: u32 = 0x000F_001F;

    const MAGIC: u32 = 0x5348_4D42;
    const VERSION: u32 = 1;
    const CAPACITY: u64 = 16;
    const INPUT_READY: u32 = 1;
    const INPUT_EMPTY: u32 = 0;
    const OUTPUT_READY: u32 = 1;
    const OUTPUT_EMPTY: u32 = 0;
    const INPUT_BASE: usize = 64;
    const INPUT_SLOT_SIZE: usize = 128;
    const OUTPUT_BASE: usize = 64 + 16 * INPUT_SLOT_SIZE;
    const OUTPUT_SLOT_SIZE: usize = 384;
    const OUTPUT_VALUES_OFFSET: usize = 64;
    const OUTPUT_METADATA_OFFSET: usize = 320;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileW(
            name: *const u16,
            access: u32,
            share: u32,
            security: *mut c_void,
            creation: u32,
            flags: u32,
            template: Handle,
        ) -> Handle;
        fn CreateFileMappingW(
            file: Handle,
            security: *mut c_void,
            protection: u32,
            size_high: u32,
            size_low: u32,
            name: *const u16,
        ) -> Handle;
        fn MapViewOfFile(
            mapping: Handle,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(address: *const c_void) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
    }

    struct Mapping {
        file: Handle,
        mapping: Handle,
        view: *mut u8,
    }

    impl Mapping {
        fn open(path: &str) -> Result<Self, String> {
            use std::os::windows::ffi::OsStrExt;
            let wide: Vec<u16> = std::ffi::OsStr::new(path)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            unsafe {
                let file = CreateFileW(
                    wide.as_ptr(),
                    GENERIC_READ | GENERIC_WRITE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    ptr::null_mut(),
                    OPEN_EXISTING,
                    0,
                    ptr::null_mut(),
                );
                if file as isize == -1 {
                    return Err(format!("CreateFileW failed: {}", std::io::Error::last_os_error()));
                }
                let mapping = CreateFileMappingW(
                    file,
                    ptr::null_mut(),
                    PAGE_READWRITE,
                    0,
                    0,
                    ptr::null(),
                );
                if mapping.is_null() {
                    CloseHandle(file);
                    return Err(format!("CreateFileMappingW failed: {}", std::io::Error::last_os_error()));
                }
                let view = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0).cast::<u8>();
                if view.is_null() {
                    CloseHandle(mapping);
                    CloseHandle(file);
                    return Err(format!("MapViewOfFile failed: {}", std::io::Error::last_os_error()));
                }
                Ok(Self { file, mapping, view })
            }
        }

        fn atomic_u32(&self, offset: usize) -> &AtomicU32 {
            assert_eq!((self.view as usize + offset) % mem::align_of::<AtomicU32>(), 0);
            unsafe { &*(self.view.add(offset).cast::<AtomicU32>()) }
        }

        fn atomic_u64(&self, offset: usize) -> &AtomicU64 {
            assert_eq!((self.view as usize + offset) % mem::align_of::<AtomicU64>(), 0);
            unsafe { &*(self.view.add(offset).cast::<AtomicU64>()) }
        }
    }

    impl Drop for Mapping {
        fn drop(&mut self) {
            unsafe {
                UnmapViewOfFile(self.view.cast());
                CloseHandle(self.mapping);
                CloseHandle(self.file);
            }
        }
    }

    fn parse_json_matrix(text: &str, key: &str) -> Result<Vec<Vec<i32>>, String> {
        let key_pos = text.find(key).ok_or_else(|| format!("missing JSON key {key}"))?;
        let bytes = text.as_bytes();
        let mut pos = bytes[key_pos..]
            .iter()
            .position(|byte| *byte == b'[')
            .map(|offset| key_pos + offset)
            .ok_or_else(|| format!("missing array for {key}"))?;
        let mut depth = 0usize;
        let mut rows = Vec::new();
        let mut current = Vec::new();
        while pos < bytes.len() {
            match bytes[pos] {
                b'[' => {
                    depth += 1;
                    if depth == 2 {
                        current = Vec::new();
                    }
                }
                b']' => {
                    if depth == 2 {
                        rows.push(core::mem::take(&mut current));
                    }
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        break;
                    }
                }
                b'-' | b'0'..=b'9' if depth == 2 => {
                    let start = pos;
                    pos += 1;
                    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                        pos += 1;
                    }
                    let value = text[start..pos]
                        .parse::<i32>()
                        .map_err(|error| error.to_string())?;
                    current.push(value);
                    continue;
                }
                _ => {}
            }
            pos += 1;
        }
        if rows.is_empty() {
            return Err(format!("empty JSON matrix {key}"));
        }
        Ok(rows)
    }

    fn wait_until<T, F: Fn() -> Option<T>>(timeout: Duration, poll: F) -> Result<T, String> {
        let start = Instant::now();
        loop {
            if let Some(value) = poll() {
                return Ok(value);
            }
            if start.elapsed() > timeout {
                return Err("timed out waiting for guest mailbox transition".to_string());
            }
            std::hint::spin_loop();
        }
    }

    pub fn run() -> Result<(), String> {
        let mut args = std::env::args().skip(1);
        let mailbox_path = args.next().ok_or("usage: host_injector <shm_mailbox.bin> <expected.json>")?;
        let expected_path = args.next().ok_or("usage: host_injector <shm_mailbox.bin> <expected.json>")?;
        let expected_text = fs::read_to_string(expected_path).map_err(|error| error.to_string())?;
        let inputs = parse_json_matrix(&expected_text, "test_batch_i8")?;
        let outputs = parse_json_matrix(&expected_text, "test_batch_output_i32")?;
        if inputs.len() != outputs.len() || inputs.is_empty() || inputs.len() > CAPACITY as usize {
            return Err("expected batch dimensions exceed mailbox capacity".to_string());
        }
        if inputs.iter().any(|row| row.len() != 64) || outputs.iter().any(|row| row.len() != 16) {
            return Err("expected JSON vectors must be [64] inputs and [16] outputs".to_string());
        }

        let mapping = Mapping::open(&mailbox_path)?;
        wait_until(Duration::from_secs(30), || unsafe {
            let magic = ptr::read_volatile(mapping.view.cast::<u32>());
            let version = ptr::read_volatile(mapping.view.add(4).cast::<u32>());
            (magic == MAGIC && version == VERSION).then_some(())
        })?;
        fence(Ordering::Acquire);
        println!("[HOST IPC]: mailbox ready magic=0x{MAGIC:08X} version={VERSION}, capacity={CAPACITY}");

        let mut latencies = Vec::with_capacity(inputs.len());
        for (frame_index, (input, expected)) in inputs.iter().zip(outputs.iter()).enumerate() {
            if input.iter().any(|value| !(-128..=127).contains(value)) {
                return Err(format!("input frame {frame_index} contains a non-int8 value"));
            }

            let input_head = mapping.atomic_u64(8);
            let input_tail = mapping.atomic_u64(16);
            let sequence = input_head.load(Ordering::Relaxed);
            let tail = input_tail.load(Ordering::Acquire);
            if sequence.wrapping_sub(tail) >= CAPACITY {
                return Err("input mailbox unexpectedly full".to_string());
            }
            let input_slot_offset = INPUT_BASE + (sequence as usize % CAPACITY as usize) * INPUT_SLOT_SIZE;
            let state = mapping.atomic_u32(input_slot_offset);
            if state.load(Ordering::Acquire) != INPUT_EMPTY {
                return Err("input slot is not empty".to_string());
            }

            let frame_start = Instant::now();
            unsafe {
                for (index, value) in input.iter().enumerate() {
                    ptr::write_volatile(mapping.view.add(input_slot_offset + 64 + index), *value as i8 as u8);
                }
            }
            // Guest T0 is measured at READY observation; the host timestamp is wall-clock only.
            mapping.atomic_u64(input_slot_offset + 8).store(0, Ordering::Relaxed);
            state.store(INPUT_READY, Ordering::Release);
            input_head.store(sequence.wrapping_add(1), Ordering::Release);

            let (output_sequence, output_slot_offset) = wait_until(Duration::from_secs(20), || {
                let output_tail = mapping.atomic_u64(32).load(Ordering::Relaxed);
                let output_head = mapping.atomic_u64(24).load(Ordering::Acquire);
                if output_tail == output_head {
                    return None;
                }
                let offset = OUTPUT_BASE + (output_tail as usize % CAPACITY as usize) * OUTPUT_SLOT_SIZE;
                let output_state = mapping.atomic_u32(offset);
                (output_state.load(Ordering::Acquire) == OUTPUT_READY)
                    .then_some((output_tail, offset))
            })?;

            let output_dim = unsafe { ptr::read_volatile(mapping.view.add(output_slot_offset + OUTPUT_METADATA_OFFSET).cast::<u32>()) };
            if output_dim != 16 {
                return Err(format!("guest output_dim={output_dim}, expected 16"));
            }
            let mut actual_bytes = [0u8; 64];
            let mut expected_bytes = [0u8; 64];
            for index in 0..16 {
                let actual = unsafe {
                    ptr::read_unaligned(mapping.view.add(output_slot_offset + OUTPUT_VALUES_OFFSET + index * 4).cast::<i32>())
                };
                if actual != expected[index] {
                    return Err(format!("frame {frame_index} output {index}: guest={actual}, expected={}", expected[index]));
                }
                actual_bytes[index * 4..index * 4 + 4].copy_from_slice(&actual.to_le_bytes());
                expected_bytes[index * 4..index * 4 + 4].copy_from_slice(&expected[index].to_le_bytes());
            }
            if actual_bytes != expected_bytes {
                return Err(format!("frame {frame_index} output bytes differ from expected bytes"));
            }

            mapping.atomic_u32(output_slot_offset).store(OUTPUT_EMPTY, Ordering::Release);
            mapping.atomic_u64(32).store(output_sequence.wrapping_add(1), Ordering::Release);
            latencies.push(frame_start.elapsed());
            println!("[HOST IPC]: frame={frame_index} parity=match round_trip_us={:.3}", latencies.last().unwrap().as_secs_f64() * 1_000_000.0);
        }

        latencies.sort_unstable();
        let median = latencies[latencies.len() / 2].as_secs_f64() * 1_000_000.0;
        println!(
            "[HOST IPC]: frames={} parity=100% latency_us_min={:.3} median={:.3} max={:.3}",
            latencies.len(),
            latencies[0].as_secs_f64() * 1_000_000.0,
            median,
            latencies[latencies.len() - 1].as_secs_f64() * 1_000_000.0
        );
        Ok(())
    }
}

fn main() {
    if let Err(error) = windows_host::run() {
        eprintln!("[HOST IPC ERROR]: {error}");
        std::process::exit(1);
    }
}
