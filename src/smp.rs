//! Minimal xAPIC INIT/SIPI startup for MADT-discovered application processors.
//!
//! The BSP parses ACPI before ExitBootServices, reserves a low trampoline page,
//! and starts only xAPIC-addressable processors. APs switch to the active custom
//! CR3, increment a fixed atomic rendezvous counter, and park in a PAUSE loop.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use uefi::table::SystemTable;

pub const MAX_APPLICATION_PROCESSORS: usize = 64;
const MAX_XAPIC_ID: usize = 255;
pub const CORE_TYPE_UNKNOWN: u8 = 0xFF;
pub const CORE_TYPE_EFFICIENCY: u8 = 0x20;
pub const CORE_TYPE_PERFORMANCE: u8 = 0x40;
const NO_MONITOR_APIC_ID: u32 = u32::MAX;
const UPDATE_IDLE: u32 = 0;
const UPDATE_QUEUED: u32 = 1;
const UPDATE_RUNNING: u32 = 2;
const UPDATE_SUCCEEDED: u32 = 3;
const UPDATE_FAILED: u32 = 4;
const UPDATE_PREPARING: u32 = 5;
const TRAMPOLINE_PHYSICAL_ADDRESS: u64 = 0x8000;
const PAGE_SIZE: usize = 4096;
const AP_STACK_SIZE: usize = 32 * 1024;
const IA32_APIC_BASE: u32 = 0x1B;
const APIC_BASE_ENABLE: u64 = 1 << 11;
const APIC_X2APIC_ENABLE: u64 = 1 << 10;
const LAPIC_SVR: usize = 0xF0;
const LAPIC_ICR_LOW: usize = 0x300;
const LAPIC_ICR_HIGH: usize = 0x310;
const ICR_DELIVERY_PENDING: u32 = 1 << 12;
const INIT_ASSERT_LEVEL: u32 = 0x0000_C500;
const INIT_DEASSERT_LEVEL: u32 = 0x0000_8500;
const STARTUP_DELIVERY: u32 = 0x0000_0600;
const ACPI_RSDP1_GUID: uefi::Guid = uefi::guid!("eb9d2d30-2d88-11d3-9a16-0090273fc14d");
const ACPI_RSDP2_GUID: uefi::Guid = uefi::guid!("8868e871-e4f1-11d3-bc22-0080c73c8881");

core::arch::global_asm!(
    ".section .rdata,\"dr\"",
    ".balign 16",
    ".global __ap_trampoline_start",
    "__ap_trampoline_start:",
    ".set __ap_gdtr_offset, __ap_gdtr - __ap_trampoline_start",
    ".set __ap_protected_far_offset, __ap_protected_far_pointer - __ap_trampoline_start",
    ".set __ap_long_far_phys, 0x8000 + __ap_long_far_pointer - __ap_trampoline_start",
    ".set __ap_cr3_phys, 0x8000 + __ap_cr3_slot - __ap_trampoline_start",
    ".set __ap_stacks_phys, 0x8000 + __ap_stacks_slot - __ap_trampoline_start",
    ".set __ap_online_phys, 0x8000 + __ap_online_slot - __ap_trampoline_start",
    ".set __ap_park_phys, 0x8000 + __ap_park_slot - __ap_trampoline_start",
    ".set __ap_core_types_phys, 0x8000 + __ap_core_types_slot - __ap_trampoline_start",
    ".set __ap_monitor_id_phys, 0x8000 + __ap_monitor_id_slot - __ap_trampoline_start",
    ".set __ap_monitor_entry_phys, 0x8000 + __ap_monitor_entry_slot - __ap_trampoline_start",
    ".set __ap_gdt_phys, 0x8000 + __ap_gdt - __ap_trampoline_start",
    ".code16",
    "cli",
    "cld",
    "xor ax, ax",
    "mov ds, ax",
    "mov es, ax",
    "mov ss, ax",
    "mov sp, 0x7000",
    "lgdt cs:[__ap_gdtr_offset]",
    "mov eax, cr0",
    "or eax, 1",
    "mov cr0, eax",
    ".byte 0x2e, 0xff, 0x2e",
    ".word __ap_protected_far_offset",
    ".code32",
    "__ap_protected:",
    "mov ax, 0x10",
    "mov ds, ax",
    "mov es, ax",
    "mov ss, ax",
    "mov esp, 0x7000",
    "mov eax, cr4",
    "or eax, 0x20",
    "mov cr4, eax",
    "mov eax, dword ptr [__ap_cr3_phys]",
    "mov cr3, eax",
    "mov ecx, 0xC0000080",
    "rdmsr",
    "or eax, 0x100",
    "wrmsr",
    "mov eax, cr0",
    "or eax, 0x80000000",
    "mov cr0, eax",
    ".byte 0xff, 0x2d",
    ".long __ap_long_far_phys",
    ".code64",
    "__ap_long_mode:",
    "mov ax, 0x10",
    "mov ds, ax",
    "mov es, ax",
    "mov ss, ax",
    "mov eax, 1",
    "cpuid",
    "shr ebx, 24",
    "and ebx, 0xff",
    "mov r9d, ebx",
    "xor eax, eax",
    "cpuid",
    "cmp eax, 0x1a",
    "jb __ap_core_uniform",
    "mov eax, 7",
    "xor ecx, ecx",
    "cpuid",
    "bt edx, 15",
    "jnc __ap_core_uniform",
    "mov eax, 0x1a",
    "xor ecx, ecx",
    "cpuid",
    "shr eax, 24",
    "jmp __ap_core_store",
    "__ap_core_uniform:",
    "xor eax, eax",
    "__ap_core_store:",
    "mov rdx, qword ptr [__ap_core_types_phys]",
    "mov byte ptr [rdx + r9], al",
    "mov rax, qword ptr [__ap_stacks_phys]",
    "mov ebx, r9d",
    "shl rbx, 15",
    "add rax, rbx",
    "add rax, 32768",
    "mov rsp, rax",
    "mov rax, qword ptr [__ap_online_phys]",
    "lock inc dword ptr [rax]",
    "mov rax, qword ptr [__ap_park_phys]",
    "__ap_park:",
    "pause",
    "cmp dword ptr [rax], 0",
    "je __ap_park",
    "mov rdx, qword ptr [__ap_monitor_id_phys]",
    "cmp dword ptr [rdx], r9d",
    "jne __ap_park",
    "mov rax, qword ptr [__ap_monitor_entry_phys]",
    "sub rsp, 32",
    "call rax",
    "jmp __ap_park",
    ".code16",
    ".balign 8",
    "__ap_gdtr:",
    ".word __ap_gdt_end - __ap_gdt - 1",
    ".long __ap_gdt_phys",
    "__ap_gdt:",
    ".quad 0",
    ".quad 0x00cf9a000000ffff",
    ".quad 0x00cf92000000ffff",
    ".quad 0x00af9a000000ffff",
    "__ap_gdt_end:",
    ".balign 4",
    "__ap_protected_far_pointer:",
    ".word 0x8000 + __ap_protected - __ap_trampoline_start",
    ".word 0x08",
    "__ap_long_far_pointer:",
    ".long 0x8000 + __ap_long_mode - __ap_trampoline_start",
    ".word 0x18",
    ".balign 8",
    ".global __ap_cr3_slot",
    "__ap_cr3_slot:",
    ".long 0",
    ".balign 8",
    ".global __ap_stacks_slot",
    "__ap_stacks_slot:",
    ".quad 0",
    ".global __ap_online_slot",
    "__ap_online_slot:",
    ".quad 0",
    ".global __ap_park_slot",
    "__ap_park_slot:",
    ".quad 0",
    ".global __ap_core_types_slot",
    "__ap_core_types_slot:",
    ".quad 0",
    ".global __ap_monitor_id_slot",
    "__ap_monitor_id_slot:",
    ".quad 0",
    ".global __ap_monitor_entry_slot",
    "__ap_monitor_entry_slot:",
    ".quad 0",
    ".global __ap_trampoline_end",
    "__ap_trampoline_end:",
);

unsafe extern "C" {
    static __ap_trampoline_start: u8;
    static __ap_trampoline_end: u8;
    static __ap_cr3_slot: u8;
    static __ap_stacks_slot: u8;
    static __ap_online_slot: u8;
    static __ap_park_slot: u8;
    static __ap_core_types_slot: u8;
    static __ap_monitor_id_slot: u8;
    static __ap_monitor_entry_slot: u8;
}

#[repr(C, align(4096))]
#[derive(Clone, Copy)]
struct ApStack([u8; AP_STACK_SIZE]);

impl ApStack {
    const ZERO: Self = Self([0; AP_STACK_SIZE]);
}

#[repr(C, align(4096))]
struct ApStacks([ApStack; MAX_XAPIC_ID + 1]);

static mut AP_STACKS: ApStacks = ApStacks([const { ApStack::ZERO }; MAX_XAPIC_ID + 1]);
static AP_ONLINE_COUNT: AtomicU32 = AtomicU32::new(0);
static AP_PARK_RELEASE: AtomicU32 = AtomicU32::new(0);
static AP_MONITOR_ID: AtomicU32 = AtomicU32::new(NO_MONITOR_APIC_ID);
static AP_MONITOR_SAMPLES: AtomicU32 = AtomicU32::new(0);
static AP_MONITOR_QUEUE_DEPTH: AtomicU32 = AtomicU32::new(0);
static UPDATE_REQUEST_LBA: AtomicU64 = AtomicU64::new(0);
static UPDATE_REQUEST_STATE: AtomicU32 = AtomicU32::new(UPDATE_IDLE);
static mut AP_CORE_TYPES: [u8; MAX_XAPIC_ID + 1] = [CORE_TYPE_UNKNOWN; MAX_XAPIC_ID + 1];

#[derive(Clone, Copy)]
pub struct SmpTopology {
    pub madt_found: bool,
    pub apic_available: bool,
    pub x2apic_mode: bool,
    pub bsp_apic_id: u32,
    pub lapic_base: u64,
    pub madt_processor_count: usize,
    pub apic_ids: [u32; MAX_APPLICATION_PROCESSORS],
    pub ap_count: usize,
    pub trampoline_address: u64,
    pub trampoline_ready: bool,
    pub hybrid_detected: bool,
    pub bsp_core_type: u8,
    pub ap_core_types: [u8; MAX_APPLICATION_PROCESSORS],
    pub online_ap_count: usize,
    pub monitor_apic_id: u32,
}

impl SmpTopology {
    pub const fn empty() -> Self {
        Self {
            madt_found: false,
            apic_available: false,
            x2apic_mode: false,
            bsp_apic_id: 0,
            lapic_base: 0xFEE0_0000,
            madt_processor_count: 0,
            apic_ids: [0; MAX_APPLICATION_PROCESSORS],
            ap_count: 0,
            trampoline_address: 0,
            trampoline_ready: false,
            hybrid_detected: false,
            bsp_core_type: CORE_TYPE_UNKNOWN,
            ap_core_types: [CORE_TYPE_UNKNOWN; MAX_APPLICATION_PROCESSORS],
            online_ap_count: 0,
            monitor_apic_id: NO_MONITOR_APIC_ID,
        }
    }
}

fn read_u32(address: usize, offset: usize) -> u32 {
    unsafe { core::ptr::read_unaligned((address + offset) as *const u32) }
}

fn read_u64(address: usize, offset: usize) -> u64 {
    unsafe { core::ptr::read_unaligned((address + offset) as *const u64) }
}

fn current_hybrid_core_type() -> (bool, u8) {
    let basic = core::arch::x86_64::__cpuid(0);
    let is_intel = basic.ebx == 0x756E_6547 && basic.edx == 0x496E_6569 && basic.ecx == 0x6C65_746E;
    if !is_intel || basic.eax < 7 {
        return (false, CORE_TYPE_UNKNOWN);
    }
    let structured = core::arch::x86_64::__cpuid_count(7, 0);
    let hybrid = structured.edx & (1 << 15) != 0;
    if !hybrid {
        return (false, 0);
    }
    if basic.eax < 0x1A {
        return (true, CORE_TYPE_UNKNOWN);
    }
    let leaf = core::arch::x86_64::__cpuid(0x1A);
    (true, (leaf.eax >> 24) as u8)
}

fn checksum_valid(address: usize, length: usize) -> bool {
    if address == 0 || length == 0 || length > 1024 * 1024 {
        return false;
    }
    let mut sum = 0u8;
    let mut offset = 0usize;
    while offset < length {
        sum =
            sum.wrapping_add(unsafe { core::ptr::read_volatile((address + offset) as *const u8) });
        offset += 1;
    }
    sum == 0
}

fn find_rsdp(system_table: &SystemTable<uefi::table::Boot>) -> Option<(usize, bool)> {
    let mut rsdp2 = None;
    let mut rsdp1 = None;
    for entry in system_table.config_table() {
        if entry.guid == ACPI_RSDP2_GUID {
            rsdp2 = Some(entry.address as usize);
        } else if entry.guid == ACPI_RSDP1_GUID {
            rsdp1 = Some(entry.address as usize);
        }
    }

    if let Some(address) = rsdp2 {
        if address != 0
            && unsafe { core::slice::from_raw_parts(address as *const u8, 8) } == b"RSD PTR "
            && checksum_valid(address, 20)
        {
            let length = read_u32(address, 20) as usize;
            let xsdt = read_u64(address, 24) as usize;
            if length >= 36 && length <= 4096 && checksum_valid(address, length) && xsdt != 0 {
                return Some((xsdt, true));
            }
        }
    }
    let address = rsdp1?;
    if address != 0
        && unsafe { core::slice::from_raw_parts(address as *const u8, 8) } == b"RSD PTR "
        && checksum_valid(address, 20)
    {
        let rsdt = read_u32(address, 16) as usize;
        if rsdt != 0 {
            return Some((rsdt, false));
        }
    }
    None
}

fn find_madt(system_table: &SystemTable<uefi::table::Boot>) -> Option<(usize, usize)> {
    find_acpi_table(system_table, b"APIC").filter(|(_, length)| *length >= 44)
}

/// Find a checksum-validated ACPI table by its four-byte signature.
pub fn find_acpi_table(
    system_table: &SystemTable<uefi::table::Boot>,
    signature: &[u8; 4],
) -> Option<(usize, usize)> {
    let (root, xsdt) = find_rsdp(system_table)?;
    let root_len = read_u32(root, 4) as usize;
    let entry_size = if xsdt { 8 } else { 4 };
    if root_len < 36 || root_len > 1024 * 1024 || (root_len - 36) % entry_size != 0 {
        return None;
    }
    let expected_root_signature = if xsdt { b"XSDT" } else { b"RSDT" };
    if unsafe { core::slice::from_raw_parts(root as *const u8, 4) } != expected_root_signature {
        return None;
    }
    if !checksum_valid(root, root_len) {
        return None;
    }
    let mut offset = 36usize;
    while offset + entry_size <= root_len {
        let table = if xsdt {
            read_u64(root, offset) as usize
        } else {
            read_u32(root, offset) as usize
        };
        if table != 0 && unsafe { core::slice::from_raw_parts(table as *const u8, 4) } == signature
        {
            let length = read_u32(table, 4) as usize;
            if length >= 36 && length <= 1024 * 1024 && checksum_valid(table, length) {
                return Some((table, length));
            }
        }
        offset += entry_size;
    }
    None
}

/// Find the ACPI MADT and collect enabled xAPIC-addressable secondary CPUs.
pub fn discover(system_table: &SystemTable<uefi::table::Boot>) -> SmpTopology {
    let mut topology = SmpTopology::empty();
    let (hybrid_detected, bsp_core_type) = current_hybrid_core_type();
    topology.hybrid_detected = hybrid_detected;
    topology.bsp_core_type = bsp_core_type;
    let cpuid0 = core::arch::x86_64::__cpuid(0);
    if cpuid0.eax < 1 {
        return topology;
    }
    let cpuid1 = core::arch::x86_64::__cpuid(1);
    topology.bsp_apic_id = cpuid1.ebx >> 24;
    topology.apic_available = cpuid1.edx & (1 << 9) != 0;
    if topology.apic_available {
        let apic_base_msr = unsafe { read_msr(IA32_APIC_BASE) };
        topology.x2apic_mode = apic_base_msr & APIC_X2APIC_ENABLE != 0;
        let msr_base = apic_base_msr & 0x0000_000F_FFFF_F000;
        if msr_base != 0 {
            topology.lapic_base = msr_base;
        }
    }

    let Some((madt, length)) = find_madt(system_table) else {
        return topology;
    };
    topology.madt_found = true;
    let madt_lapic = read_u32(madt, 36) as u64;
    if madt_lapic != 0 {
        topology.lapic_base = madt_lapic;
    }

    let mut offset = 44usize;
    while offset + 2 <= length {
        let entry_type = unsafe { core::ptr::read_volatile((madt + offset) as *const u8) };
        let entry_length =
            unsafe { core::ptr::read_volatile((madt + offset + 1) as *const u8) } as usize;
        if entry_length < 2 || offset + entry_length > length {
            break;
        }
        let (apic_id, flags) = match entry_type {
            0 if entry_length >= 8 => {
                let apic_id =
                    unsafe { core::ptr::read_volatile((madt + offset + 3) as *const u8) } as u32;
                (Some(apic_id), read_u32(madt + offset, 4))
            }
            9 if entry_length >= 16 => {
                (Some(read_u32(madt + offset, 4)), read_u32(madt + offset, 8))
            }
            5 if entry_length >= 12 => {
                topology.lapic_base = read_u64(madt + offset, 4);
                (None, 0)
            }
            _ => (None, 0),
        };
        if let Some(apic_id) = apic_id {
            // Only processors marked enabled may be started directly. An
            // online-capable but disabled CPU needs ACPI AML _STA/_MAT handling.
            if flags & 0x1 != 0 {
                topology.madt_processor_count += 1;
                if apic_id != topology.bsp_apic_id
                    && apic_id <= MAX_XAPIC_ID as u32
                    && topology.ap_count < MAX_APPLICATION_PROCESSORS
                    && !topology.apic_ids[..topology.ap_count].contains(&apic_id)
                {
                    topology.apic_ids[topology.ap_count] = apic_id;
                    topology.ap_count += 1;
                }
            }
        }
        offset += entry_length;
    }

    if !topology.apic_available || topology.x2apic_mode {
        topology.ap_count = 0;
    }
    topology
}

/// Reserve and populate a SIPI page while Boot Services are still available.
pub fn prepare_trampoline(topology: &mut SmpTopology) -> bool {
    if topology.ap_count == 0
        || !topology.madt_found
        || !topology.apic_available
        || topology.x2apic_mode
    {
        return false;
    }
    let pages = match uefi::boot::allocate_pages(
        uefi::boot::AllocateType::Address(TRAMPOLINE_PHYSICAL_ADDRESS),
        uefi::table::boot::MemoryType::LOADER_DATA,
        1,
    ) {
        Ok(address) => address,
        Err(_) => return false,
    };
    let start = core::ptr::addr_of!(__ap_trampoline_start) as usize;
    let end = core::ptr::addr_of!(__ap_trampoline_end) as usize;
    let length = end.saturating_sub(start);
    if length == 0 || length > PAGE_SIZE {
        return false;
    }
    unsafe {
        let destination = pages.as_ptr().cast::<u8>();
        core::ptr::write_bytes(destination, 0, PAGE_SIZE);
        core::ptr::copy_nonoverlapping(start as *const u8, destination, length);
    }
    topology.trampoline_address = TRAMPOLINE_PHYSICAL_ADDRESS;
    topology.trampoline_ready = true;
    true
}

/// Send INIT/SIPI sequences, wait for AP check-ins, and leave APs parked.
/// Must be called on the BSP after the custom CR3 identity map is active.
pub unsafe fn start_application_processors(topology: &mut SmpTopology) -> u32 {
    if topology.ap_count == 0 || !topology.trampoline_ready || topology.x2apic_mode {
        crate::serial_println!(
            "[SMP]: 0/{} Application Processors awakened and parked (MADT={}, x2APIC={}, trampoline={}).",
            topology.ap_count,
            topology.madt_found,
            topology.x2apic_mode,
            topology.trampoline_ready
        );
        return 0;
    }

    let cr3 = read_cr3();
    if cr3 > u32::MAX as u64 {
        crate::serial_println!(
            "[SMP WARNING]: custom CR3 is above 4 GiB; AP trampoline cannot load it."
        );
        return 0;
    }
    let trampoline = topology.trampoline_address as *mut u8;
    write_u32(
        trampoline,
        symbol_offset(core::ptr::addr_of!(__ap_cr3_slot)),
        cr3 as u32,
    );
    write_u64(
        trampoline,
        symbol_offset(core::ptr::addr_of!(__ap_stacks_slot)),
        core::ptr::addr_of_mut!(AP_STACKS) as u64,
    );
    write_u64(
        trampoline,
        symbol_offset(core::ptr::addr_of!(__ap_online_slot)),
        core::ptr::addr_of!(AP_ONLINE_COUNT) as u64,
    );
    write_u64(
        trampoline,
        symbol_offset(core::ptr::addr_of!(__ap_park_slot)),
        core::ptr::addr_of!(AP_PARK_RELEASE) as u64,
    );
    write_u64(
        trampoline,
        symbol_offset(core::ptr::addr_of!(__ap_core_types_slot)),
        core::ptr::addr_of_mut!(AP_CORE_TYPES) as u64,
    );
    write_u64(
        trampoline,
        symbol_offset(core::ptr::addr_of!(__ap_monitor_id_slot)),
        core::ptr::addr_of!(AP_MONITOR_ID) as u64,
    );
    write_u64(
        trampoline,
        symbol_offset(core::ptr::addr_of!(__ap_monitor_entry_slot)),
        crate::ap_ring_monitor_entry as *const () as u64,
    );
    let core_types = core::ptr::addr_of_mut!(AP_CORE_TYPES).cast::<u8>();
    let mut apic_id = 0usize;
    while apic_id <= MAX_XAPIC_ID {
        core::ptr::write_volatile(core_types.add(apic_id), CORE_TYPE_UNKNOWN);
        apic_id += 1;
    }
    AP_ONLINE_COUNT.store(0, Ordering::Relaxed);
    AP_PARK_RELEASE.store(0, Ordering::Relaxed);
    AP_MONITOR_ID.store(NO_MONITOR_APIC_ID, Ordering::Relaxed);
    AP_MONITOR_SAMPLES.store(0, Ordering::Relaxed);
    AP_MONITOR_QUEUE_DEPTH.store(0, Ordering::Relaxed);

    let apic_base_msr = read_msr(IA32_APIC_BASE);
    if apic_base_msr & APIC_X2APIC_ENABLE != 0 {
        crate::serial_println!(
            "[SMP WARNING]: x2APIC mode became active; xAPIC SIPI path skipped."
        );
        return 0;
    }
    if apic_base_msr & APIC_BASE_ENABLE == 0 {
        write_msr(IA32_APIC_BASE, apic_base_msr | APIC_BASE_ENABLE);
    }
    let lapic = topology.lapic_base as *mut u32;
    let svr = lapic.add(LAPIC_SVR / 4);
    let svr_value = core::ptr::read_volatile(svr);
    core::ptr::write_volatile(svr, (svr_value & !0xFF) | 0x100 | 0xFF);

    let frequency = crate::timer::tsc_frequency_hz();
    if frequency == 0 {
        crate::serial_println!("[SMP WARNING]: TSC calibration unavailable; AP startup skipped.");
        return 0;
    }
    for index in 0..topology.ap_count {
        let apic_id = topology.apic_ids[index];
        let icr_high = lapic.add(LAPIC_ICR_HIGH / 4);
        let icr_low = lapic.add(LAPIC_ICR_LOW / 4);

        wait_icr_idle(icr_low);
        core::ptr::write_volatile(icr_high, apic_id << 24);
        core::ptr::write_volatile(icr_low, INIT_ASSERT_LEVEL);
        wait_icr_idle(icr_low);
        delay_microseconds(frequency, 10_000);

        core::ptr::write_volatile(icr_high, apic_id << 24);
        core::ptr::write_volatile(icr_low, INIT_DEASSERT_LEVEL);
        wait_icr_idle(icr_low);
        delay_microseconds(frequency, 200);

        let vector = (topology.trampoline_address >> 12) as u32;
        core::ptr::write_volatile(icr_high, apic_id << 24);
        core::ptr::write_volatile(icr_low, STARTUP_DELIVERY | vector);
        wait_icr_idle(icr_low);
        delay_microseconds(frequency, 200);

        core::ptr::write_volatile(icr_high, apic_id << 24);
        core::ptr::write_volatile(icr_low, STARTUP_DELIVERY | vector);
        wait_icr_idle(icr_low);
    }

    let timeout_cycles = frequency.saturating_div(2);
    let start = crate::timer::read_tsc();
    while AP_ONLINE_COUNT.load(Ordering::Acquire) < topology.ap_count as u32
        && crate::timer::read_tsc().wrapping_sub(start) < timeout_cycles
    {
        core::hint::spin_loop();
    }
    let online = AP_ONLINE_COUNT.load(Ordering::Acquire);
    let core_types = core::ptr::addr_of!(AP_CORE_TYPES).cast::<u8>();
    let mut index = 0usize;
    let mut selected_monitor = NO_MONITOR_APIC_ID;
    while index < topology.ap_count {
        let id = topology.apic_ids[index];
        let core_type = core::ptr::read_volatile(core_types.add(id as usize));
        topology.ap_core_types[index] = core_type;
        if core_type != CORE_TYPE_UNKNOWN {
            topology.online_ap_count += 1;
        }
        if selected_monitor == NO_MONITOR_APIC_ID
            && (!topology.hybrid_detected || core_type == CORE_TYPE_EFFICIENCY)
            && core_type != CORE_TYPE_UNKNOWN
        {
            selected_monitor = id;
        }
        index += 1;
    }
    if selected_monitor == NO_MONITOR_APIC_ID {
        index = 0;
        while index < topology.ap_count {
            if topology.ap_core_types[index] != CORE_TYPE_UNKNOWN {
                selected_monitor = topology.apic_ids[index];
                break;
            }
            index += 1;
        }
    }
    topology.monitor_apic_id = selected_monitor;
    if selected_monitor != NO_MONITOR_APIC_ID {
        AP_MONITOR_ID.store(selected_monitor, Ordering::Release);
        AP_PARK_RELEASE.store(1, Ordering::Release);
    }
    crate::serial_println!(
        "[SMP]: {}/{} Application Processors awakened and parked",
        online,
        topology.ap_count
    );
    online
}

pub fn record_ring_monitor_sample(queue_depth: usize) {
    AP_MONITOR_QUEUE_DEPTH.store(queue_depth.min(u32::MAX as usize) as u32, Ordering::Relaxed);
    AP_MONITOR_SAMPLES.fetch_add(1, Ordering::Relaxed);
}

pub fn ring_monitor_stats() -> (u32, u32, u32) {
    (
        AP_MONITOR_ID.load(Ordering::Acquire),
        AP_MONITOR_SAMPLES.load(Ordering::Acquire),
        AP_MONITOR_QUEUE_DEPTH.load(Ordering::Acquire),
    )
}

pub fn log_ring_monitor_status() {
    let (monitor_id, samples, queue_depth) = ring_monitor_stats();
    if monitor_id != NO_MONITOR_APIC_ID {
        crate::serial_println!(
            "[TOPOLOGY AUX VERIFY]: APIC ID={}, ring_samples={}, queue_depth={}",
            monitor_id,
            samples,
            queue_depth
        );
    }
}

/// Queue one update request for the auxiliary AP. Returns false if one is active.
pub fn request_shard_update(lba: u64) -> bool {
    if AP_MONITOR_ID.load(Ordering::Acquire) == NO_MONITOR_APIC_ID {
        return false;
    }
    if UPDATE_REQUEST_STATE
        .compare_exchange(
            UPDATE_IDLE,
            UPDATE_PREPARING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        return false;
    }
    UPDATE_REQUEST_LBA.store(lba, Ordering::Relaxed);
    UPDATE_REQUEST_STATE.store(UPDATE_QUEUED, Ordering::Release);
    true
}

/// Claim a queued update on the AP exactly once.
pub fn take_shard_update_request() -> Option<u64> {
    UPDATE_REQUEST_STATE
        .compare_exchange(
            UPDATE_QUEUED,
            UPDATE_RUNNING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .ok()
        .map(|_| UPDATE_REQUEST_LBA.load(Ordering::Acquire))
}

pub fn finish_shard_update(success: bool) {
    UPDATE_REQUEST_STATE.store(
        if success {
            UPDATE_SUCCEEDED
        } else {
            UPDATE_FAILED
        },
        Ordering::Release,
    );
}

pub fn shard_update_lba() -> u64 {
    UPDATE_REQUEST_LBA.load(Ordering::Acquire)
}

pub fn shard_update_in_progress() -> bool {
    matches!(
        UPDATE_REQUEST_STATE.load(Ordering::Acquire),
        UPDATE_PREPARING | UPDATE_QUEUED | UPDATE_RUNNING
    )
}

/// Consume one completion result and return the request slot to idle.
pub fn take_shard_update_result() -> Option<bool> {
    loop {
        let state = UPDATE_REQUEST_STATE.load(Ordering::Acquire);
        let success = match state {
            UPDATE_SUCCEEDED => true,
            UPDATE_FAILED => false,
            _ => return None,
        };
        if UPDATE_REQUEST_STATE
            .compare_exchange(state, UPDATE_IDLE, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return Some(success);
        }
    }
}

/// Log the detected per-processor topology and the active core-role mapping.
pub fn log_topology(topology: &SmpTopology) {
    if !topology.hybrid_detected {
        crate::serial_println!("[TOPOLOGY]: Uniform/Symmetric");
    } else {
        let mut p_core = if topology.bsp_core_type == CORE_TYPE_PERFORMANCE {
            Some(topology.bsp_apic_id)
        } else {
            None
        };
        let mut e_core = if topology.bsp_core_type == CORE_TYPE_EFFICIENCY {
            Some(topology.bsp_apic_id)
        } else {
            None
        };
        let mut index = 0usize;
        while index < topology.ap_count {
            match topology.ap_core_types[index] {
                CORE_TYPE_PERFORMANCE if p_core.is_none() => {
                    p_core = Some(topology.apic_ids[index]);
                }
                CORE_TYPE_EFFICIENCY if e_core.is_none() => {
                    e_core = Some(topology.apic_ids[index]);
                }
                _ => {}
            }
            index += 1;
        }
        match (p_core, e_core) {
            (Some(p), Some(e)) => {
                crate::serial_println!("[TOPOLOGY]: Hybrid (P-Core={}, E-Core={})", p, e)
            }
            (Some(p), None) => {
                crate::serial_println!("[TOPOLOGY]: Hybrid (P-Core={}, E-Core=unavailable)", p)
            }
            (None, Some(e)) => {
                crate::serial_println!("[TOPOLOGY]: Hybrid (P-Core=unavailable, E-Core={})", e)
            }
            (None, None) => crate::serial_println!(
                "[TOPOLOGY]: Hybrid (P-Core=unclassified, E-Core=unclassified)"
            ),
        }
        if topology.bsp_core_type == CORE_TYPE_PERFORMANCE {
            crate::serial_println!(
                "[TOPOLOGY PINNING]: deterministic inference loop remains on BSP P-Core={}",
                topology.bsp_apic_id
            );
        } else {
            crate::serial_println!(
                "[TOPOLOGY WARNING]: BSP core_type=0x{:02X}; no safe inference migration to a P-Core is available in this firmware build.",
                topology.bsp_core_type
            );
        }
    }

    let (monitor_id, samples, queue_depth) = ring_monitor_stats();
    if monitor_id != NO_MONITOR_APIC_ID {
        let mut core_type = CORE_TYPE_UNKNOWN;
        let mut index = 0usize;
        while index < topology.ap_count {
            if topology.apic_ids[index] == monitor_id {
                core_type = topology.ap_core_types[index];
                break;
            }
            index += 1;
        }
        crate::serial_println!(
            "[TOPOLOGY AUX]: ring-monitor APIC ID={}, core_type=0x{:02X}, samples={}, queue_depth={}",
            monitor_id,
            core_type,
            samples,
            queue_depth
        );
    } else {
        crate::serial_println!(
            "[TOPOLOGY AUX]: no secondary processor available; BSP monitors the rings."
        );
    }
}

fn symbol_offset(symbol: *const u8) -> usize {
    symbol as usize - core::ptr::addr_of!(__ap_trampoline_start) as usize
}

unsafe fn write_u32(destination: *mut u8, offset: usize, value: u32) {
    core::ptr::write_unaligned(destination.add(offset).cast::<u32>(), value);
}

unsafe fn write_u64(destination: *mut u8, offset: usize, value: u64) {
    core::ptr::write_unaligned(destination.add(offset).cast::<u64>(), value);
}

unsafe fn read_cr3() -> u64 {
    let value: u64;
    core::arch::asm!("mov {}, cr3", out(reg) value, options(nomem, nostack, preserves_flags));
    value
}

unsafe fn read_msr(msr: u32) -> u64 {
    let low: u32;
    let high: u32;
    core::arch::asm!(
        "rdmsr",
        in("ecx") msr,
        out("eax") low,
        out("edx") high,
        options(nomem, nostack, preserves_flags),
    );
    ((high as u64) << 32) | low as u64
}

unsafe fn write_msr(msr: u32, value: u64) {
    core::arch::asm!(
        "wrmsr",
        in("ecx") msr,
        in("eax") value as u32,
        in("edx") (value >> 32) as u32,
        options(nomem, nostack, preserves_flags),
    );
}

unsafe fn wait_icr_idle(icr_low: *mut u32) {
    let mut retries = 1_000_000usize;
    while core::ptr::read_volatile(icr_low) & ICR_DELIVERY_PENDING != 0 && retries != 0 {
        core::hint::spin_loop();
        retries -= 1;
    }
}

fn delay_microseconds(frequency: u64, microseconds: u64) {
    let cycles =
        (frequency as u128 * microseconds as u128 / 1_000_000).min(u64::MAX as u128) as u64;
    let start = unsafe { crate::timer::read_tsc() };
    while unsafe { crate::timer::read_tsc() }.wrapping_sub(start) < cycles {
        core::hint::spin_loop();
    }
}
