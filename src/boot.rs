//! UEFI Boot Services exit, memory map acquisition, and absolute CPU sovereignty.

use core::mem::MaybeUninit;
use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType};
use uefi::proto::media::fs::SimpleFileSystem;

/// 16 KiB statically aligned memory map buffer (~300+ memory descriptors).
#[repr(C, align(64))]
pub struct MmapBuffer {
    pub data: [u8; 16384],
}

pub static mut MMAP_BUFFER: MmapBuffer = MmapBuffer { data: [0u8; 16384] };

/// Reads a model shard from any UEFI FAT volume into caller-owned storage.
///
/// Handle enumeration and file contents use fixed caller buffers; this does
/// not enable Rust `alloc` or allocate a file-sized `Vec`.
pub fn read_weights_file(buffer: &mut [u8]) -> Option<usize> {
    const MAX_FILESYSTEM_HANDLES: usize = 64;
    let mut handle_storage = [MaybeUninit::uninit(); MAX_FILESYSTEM_HANDLES];
    let handles = uefi::boot::locate_handle(
        uefi::boot::SearchType::from_proto::<SimpleFileSystem>(),
        &mut handle_storage,
    )
    .ok()?;

    let paths = [
        uefi::cstr16!("\\weights.bin"),
        uefi::cstr16!("\\NEURAL_WEIGHTS\\weights.bin"),
    ];
    for handle in handles.iter().cloned() {
        let mut filesystem = match uefi::boot::open_protocol_exclusive::<SimpleFileSystem>(handle) {
            Ok(protocol) => protocol,
            Err(_) => continue,
        };
        let mut root = match filesystem.open_volume() {
            Ok(directory) => directory,
            Err(_) => continue,
        };

        for path in paths {
            let file = match root.open(path, FileMode::Read, FileAttribute::empty()) {
                Ok(file) => file,
                Err(_) => continue,
            };
            let mut file = match file.into_type() {
                Ok(FileType::Regular(file)) => file,
                _ => continue,
            };

            let mut bytes_read = 0usize;
            while bytes_read < buffer.len() {
                match file.read(&mut buffer[bytes_read..]) {
                    Ok(0) | Err(_) => break,
                    Ok(count) => bytes_read += count,
                }
            }
            if bytes_read != 0 {
                return Some(bytes_read);
            }
        }
    }
    None
}

/// Bind the mailbox to the IVSHMEM shared-memory BAR when it is large enough.
///
/// # Safety
/// The UEFI platform must provide an identity-mapped, writable, cache-coherent
/// mapping for the PCI shared-memory BAR. Firmware typically maps PCI BARs as
/// uncacheable; this function does not edit page tables after firmware setup.
pub unsafe fn bind_ivshmem_mailbox(
    device: crate::pci::PciSharedMemoryInfo,
) -> Option<core::ptr::NonNull<crate::shared_mem::SharedMailbox>> {
    let mailbox_size = core::mem::size_of::<crate::shared_mem::SharedMailbox>() as u64;
    if device.shared_memory_size < mailbox_size
        || device.shared_memory_phys
            % core::mem::align_of::<crate::shared_mem::SharedMailbox>() as u64
            != 0
    {
        return None;
    }
    crate::shared_mem::SharedMailbox::bind_host_window(device.shared_memory_phys).ok()
}

/// Allocates and initializes a persistent-across-ExitBootServices mailbox region.
/// The returned address is guest physical RAM with an identity-mapped UEFI pointer;
/// it is not automatically mapped into the host process.
#[cfg(not(feature = "host-ipc"))]
pub fn reserve_shared_mailbox() -> uefi::Result<core::ptr::NonNull<crate::shared_mem::SharedMailbox>>
{
    let bytes = core::mem::size_of::<crate::shared_mem::SharedMailbox>();
    let pages = bytes.div_ceil(4096);
    let allocation = uefi::boot::allocate_pages(
        uefi::boot::AllocateType::AnyPages,
        uefi::table::boot::MemoryType::LOADER_DATA,
        pages,
    )?;
    let mailbox = allocation.cast::<crate::shared_mem::SharedMailbox>();
    unsafe {
        mailbox
            .as_ptr()
            .write(crate::shared_mem::SharedMailbox::new());
    }
    Ok(mailbox)
}

#[cfg(feature = "host-ipc")]
pub fn reserve_shared_mailbox(
) -> Result<core::ptr::NonNull<crate::shared_mem::SharedMailbox>, &'static str> {
    unsafe {
        crate::shared_mem::SharedMailbox::bind_host_window(
            crate::shared_mem::HOST_MAILBOX_PHYSICAL_BASE,
        )
    }
    .map_err(|_| "host-backed mailbox address is invalid or unaligned")
}

/// Information about the acquired UEFI memory map.
pub struct MemoryMapInfo {
    pub map_size: usize,
    pub map_key: usize,
    pub desc_size: usize,
    pub desc_version: u32,
    pub entry_count: usize,
}

const CPUID_TME_ACTIVATE: u32 = 1 << 13;
const CPUID_AMD_SME: u32 = 1 << 0;
const CPUID_AMD_SEV: u32 = 1 << 1;
const AMD_SYSCFG_MEM_ENCRYPTION_ENABLED: u64 = 1 << 23;
const IA32_TME_ACTIVATE: u32 = 0x982;
const AMD64_SYSCFG: u32 = 0xC001_0010;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryEncryptionStatus {
    Unsupported,
    IntelTme {
        active: bool,
        locked: bool,
        algorithm: u16,
    },
    AmdSme {
        active: bool,
        sev_supported: bool,
        c_bit_position: Option<u8>,
    },
}

impl MemoryEncryptionStatus {
    /// Strict deployments require immutable, active TME or active SME with a usable C-bit.
    pub const fn satisfies_strict_policy(self) -> bool {
        match self {
            Self::IntelTme {
                active,
                locked,
                algorithm,
            } => active && locked && (algorithm == 0 || algorithm == 1),
            Self::AmdSme {
                active,
                c_bit_position,
                ..
            } => {
                active
                    && matches!(c_bit_position, Some(position) if position >= 12 && position <= 51)
            }
            Self::Unsupported => false,
        }
    }

    /// The AMD C-bit is applied to RAM page-table entries, never Intel TME mappings.
    pub const fn amd_c_bit_mask(self) -> Option<u64> {
        match self {
            Self::AmdSme {
                active: true,
                c_bit_position: Some(position),
                ..
            } if position < 64 => Some(1u64 << position),
            _ => None,
        }
    }
}

#[inline]
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

fn decode_intel_tme(activate: u64) -> MemoryEncryptionStatus {
    MemoryEncryptionStatus::IntelTme {
        active: activate & (1 << 1) != 0,
        locked: activate & 1 != 0,
        algorithm: (activate >> 48) as u16,
    }
}

fn decode_amd_sme(
    sme_supported: bool,
    sev_supported: bool,
    sys_cfg: u64,
    cpuid_ebx: u32,
) -> MemoryEncryptionStatus {
    let c_bit = (cpuid_ebx & 0x3f) as u8;
    MemoryEncryptionStatus::AmdSme {
        active: sys_cfg & AMD_SYSCFG_MEM_ENCRYPTION_ENABLED != 0,
        sev_supported,
        c_bit_position: if sme_supported && (12..=51).contains(&c_bit) {
            Some(c_bit)
        } else {
            None
        },
    }
}

/// Probe hardware transparent memory encryption without touching unsupported MSRs.
///
/// RDMSR is issued only after the corresponding architectural CPUID capability
/// is present and the vendor/leaf range matches. Legacy CPUs and emulators that
/// do not advertise TME/SME therefore never execute these MSR accesses.
pub fn memory_encryption_status() -> MemoryEncryptionStatus {
    let basic = core::arch::x86_64::__cpuid(0);
    let mut vendor = [0u8; 12];
    vendor[..4].copy_from_slice(&basic.ebx.to_le_bytes());
    vendor[4..8].copy_from_slice(&basic.edx.to_le_bytes());
    vendor[8..].copy_from_slice(&basic.ecx.to_le_bytes());
    if &vendor == b"GenuineIntel" {
        if basic.eax < 7 {
            return MemoryEncryptionStatus::Unsupported;
        }
        let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
        if leaf7.ecx & CPUID_TME_ACTIVATE == 0 {
            return MemoryEncryptionStatus::Unsupported;
        }
        // Intel documents IA32_TME_ACTIVATE when CPUID.(EAX=7,ECX=0):ECX[13] is set.
        return decode_intel_tme(unsafe { read_msr(IA32_TME_ACTIVATE) });
    }

    if &vendor == b"AuthenticAMD" {
        let extended = core::arch::x86_64::__cpuid(0x8000_0000).eax;
        if extended < 0x8000_001f {
            return MemoryEncryptionStatus::Unsupported;
        }
        let leaf = core::arch::x86_64::__cpuid(0x8000_001f);
        let sme_supported = leaf.eax & CPUID_AMD_SME != 0;
        let sev_supported = leaf.eax & CPUID_AMD_SEV != 0;
        if !sme_supported && !sev_supported {
            return MemoryEncryptionStatus::Unsupported;
        }
        // AMD exposes SYSCFG.MemEncryptionEnabled when SME/SEV is enumerated.
        return decode_amd_sme(
            sme_supported,
            sev_supported,
            unsafe { read_msr(AMD64_SYSCFG) },
            leaf.ebx,
        );
    }

    MemoryEncryptionStatus::Unsupported
}

/// Reports CPU support for the AVX-512 kernel; OS vector state is enabled separately.
pub fn cpu_supports_avx512() -> bool {
    let (f, bw, dq, vl, xsave, _, _) = avx512_guest_features();
    f && bw && dq && vl && xsave
}

/// Reports AVX-512F, AVX-512BW, and AVX-512VPOPCNTDQ hardware support.
pub fn cpu_supports_avx512_vpopcntdq() -> bool {
    let max_leaf = core::arch::x86_64::__cpuid(0).eax;
    if max_leaf < 7 {
        return false;
    }

    let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
    let avx512f = (leaf7.ebx & (1 << 16)) != 0;
    let avx512bw = (leaf7.ebx & (1 << 30)) != 0;
    let avx512vpopcntdq = (leaf7.ecx & (1 << 14)) != 0;
    avx512f && avx512bw && avx512vpopcntdq
}

/// Reports AVX-512F and XSAVE prerequisites for enabling OS ZMM state.
/// OSXSAVE may initially be clear; `enable_avx512_os_state` enables it before XGETBV.
pub fn cpu_supports_avx512_state() -> bool {
    let (f, _, _, _, xsave, _, _) = avx512_guest_features();
    f && xsave
}

#[derive(Clone, Copy, Debug)]
pub struct L3CatCapabilities {
    pub cbm_length: u8,
    pub clos_count: u32,
}

/// Enumerates Intel L3 Cache Allocation Technology without touching CAT MSRs.
/// MSR access is deferred until after this CPUID capability gate succeeds.
pub fn l3_cat_capabilities() -> Option<L3CatCapabilities> {
    let max_leaf = core::arch::x86_64::__cpuid(0).eax;
    if max_leaf < 0x10 {
        return None;
    }
    let rdt_leaf = core::arch::x86_64::__cpuid_count(7, 0);
    if rdt_leaf.ebx & (1 << 15) == 0 {
        return None;
    }
    let allocation_leaf = core::arch::x86_64::__cpuid_count(0x10, 0);
    if allocation_leaf.ebx & (1 << 1) == 0 {
        return None;
    }
    let l3_leaf = core::arch::x86_64::__cpuid_count(0x10, 1);
    let cbm_length = ((l3_leaf.eax & 0x1F) + 1) as u8;
    let clos_count = (l3_leaf.ecx & 0xFFFF) + 1;
    if cbm_length < 2 || clos_count < 2 {
        return None;
    }
    Some(L3CatCapabilities {
        cbm_length,
        clos_count,
    })
}

/// Returns guest-visible AVX-512 CPUID bits, XSAVE/OSXSAVE, and the current XCR0 value.
pub fn avx512_guest_features() -> (bool, bool, bool, bool, bool, bool, u64) {
    let max_leaf = core::arch::x86_64::__cpuid(0).eax;
    if max_leaf < 7 {
        return (false, false, false, false, false, false, 0);
    }

    let leaf1 = core::arch::x86_64::__cpuid(1);
    let xsave = (leaf1.ecx & (1 << 26)) != 0;
    let osxsave = (leaf1.ecx & (1 << 27)) != 0;

    let leaf7 = core::arch::x86_64::__cpuid_count(7, 0);
    let avx512f = (leaf7.ebx & (1 << 16)) != 0;
    let avx512dq = (leaf7.ebx & (1 << 17)) != 0;
    let avx512bw = (leaf7.ebx & (1 << 30)) != 0;
    let avx512vl = (leaf7.ebx & (1 << 31)) != 0;
    let xcr0 = if xsave && osxsave {
        let low: u32;
        let high: u32;
        unsafe {
            core::arch::asm!(
                "xgetbv",
                in("ecx") 0u32,
                out("eax") low,
                out("edx") high,
                options(nomem, nostack, preserves_flags),
            );
        }
        ((high as u64) << 32) | low as u64
    } else {
        0
    };
    (avx512f, avx512bw, avx512dq, avx512vl, xsave, osxsave, xcr0)
}

/// Reports whether TSC is invariant and whether RDTSCP is implemented.
pub fn tsc_capabilities() -> (bool, bool) {
    let max_extended = core::arch::x86_64::__cpuid(0x8000_0000).eax;
    if max_extended < 0x8000_0007 {
        return (false, false);
    }
    let invariant_tsc = (core::arch::x86_64::__cpuid(0x8000_0007).edx & (1 << 8)) != 0;
    let rdtscp = if max_extended >= 0x8000_0001 {
        (core::arch::x86_64::__cpuid(0x8000_0001).edx & (1 << 27)) != 0
    } else {
        false
    };
    (invariant_tsc, rdtscp)
}

/// Reports HWP/turbo capability and, when supported, the architectural HWP enable state.
pub fn cpu_power_management_status() -> (bool, bool, bool) {
    let max_leaf = core::arch::x86_64::__cpuid(0).eax;
    if max_leaf < 6 {
        return (false, false, false);
    }

    let leaf6 = core::arch::x86_64::__cpuid(6);
    let turbo_supported = (leaf6.eax & (1 << 1)) != 0;
    let hwp_supported = (leaf6.eax & (1 << 7)) != 0;
    let hwp_enabled = if hwp_supported {
        let low: u32;
        let high: u32;
        unsafe {
            core::arch::asm!(
                "rdmsr",
                in("ecx") 0x770u32,
                out("eax") low,
                out("edx") high,
                options(nomem, nostack, preserves_flags),
            );
        }
        let _ = high;
        (low & 1) != 0
    } else {
        false
    };

    (hwp_supported, hwp_enabled, turbo_supported)
}

/// Retrieves the UEFI memory map into the provided buffer and attempts to exit boot services.
/// Handles potential retry loops if the map key changes between retrieval and exit.
pub unsafe fn exit_uefi_boot_services(
    image_handle: uefi::Handle,
    system_table: &mut uefi::table::SystemTable<uefi::table::Boot>,
) -> MemoryMapInfo {
    let st_raw: *const uefi_raw::table::system::SystemTable = system_table.as_ptr().cast();
    let bs_table_raw = (*st_raw).boot_services;
    let get_memory_map = (*bs_table_raw).get_memory_map;
    let exit_boot_services_fn = (*bs_table_raw).exit_boot_services;

    let buf_ptr = core::ptr::addr_of_mut!(MMAP_BUFFER.data);
    let buf_len = (*buf_ptr).len();
    let mut map_size: usize;
    let mut map_key: usize = 0;
    let mut desc_size: usize = 0;
    let mut desc_version: u32 = 0;

    // Retry loop: firmware events/timers may update the memory map between
    // get_memory_map and exit_boot_services, invalidating map_key.
    const MAX_RETRIES: usize = 5;
    let mut attempt = 0;

    while attempt < MAX_RETRIES {
        map_size = buf_len;
        let status = get_memory_map(
            &mut map_size,
            (*buf_ptr).as_mut_ptr().cast(),
            &mut map_key,
            &mut desc_size,
            &mut desc_version,
        );

        if status != uefi::Status::SUCCESS {
            panic!("Failed to retrieve UEFI memory map: {:?}", status);
        }

        let exit_status = exit_boot_services_fn(image_handle.as_ptr().cast(), map_key);
        if exit_status == uefi::Status::SUCCESS {
            let entry_count = if desc_size > 0 {
                map_size / desc_size
            } else {
                0
            };
            return MemoryMapInfo {
                map_size,
                map_key,
                desc_size,
                desc_version,
                entry_count,
            };
        }

        attempt += 1;
    }

    panic!("Exceeded maximum retries attempting to exit UEFI boot services");
}

/// Establishes absolute CPU sovereignty:
/// 1. Suppresses all hardware interrupts (`cli`).
/// 2. Verifies and enforces control registers:
///    - `CR0.PE = 1` (Protected/Long mode)
///    - `CR0.TS = 0` (Task Switched cleared)
///    - `CR4.OSFXSR = 1` (SSE enabled)
///    - `CR4.OSXSAVE = 1` (XSAVE enabled)
///    - `XCR0` holds AVX-512 state (`X87 | SSE | AVX | opmask | ZMM_Hi256 | Hi16_ZMM`)
pub unsafe fn establish_cpu_sovereignty(avx512_enabled: bool) {
    // 1. Suppress all maskable hardware interrupts immediately
    core::arch::asm!("cli", options(nomem, nostack, preserves_flags));
    let rflags: u64;
    core::arch::asm!("pushfq", "pop {}", out(reg) rflags, options(nomem, preserves_flags));
    if (rflags & (1 << 9)) != 0 {
        panic!("CPU interrupt mask verification failed after CLI");
    }

    // 2. Control Register Verification & Enforcement
    let mut cr0: u64;
    core::arch::asm!("mov {}, cr0", out(reg) cr0, options(nomem, nostack, preserves_flags));
    // Verify Protected/Long Mode (PE = 1)
    if (cr0 & 1) == 0 {
        panic!("CPU Sovereignty Violation: CR0.PE != 1");
    }
    // Clear Task Switched (TS = 0)
    core::arch::asm!("clts", options(nomem, nostack, preserves_flags));

    // Enable SSE state unconditionally; AVX state is enabled only after a guarded probe.
    let mut cr4: u64;
    core::arch::asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
    cr4 |= 1 << 9;
    if avx512_enabled {
        cr4 |= 1 << 18;
    }
    core::arch::asm!("mov cr4, {}", in(reg) cr4, options(nomem, nostack, preserves_flags));

    if avx512_enabled {
        let xcr0_lo: u32;
        let xcr0_hi: u32;
        core::arch::asm!(
            "xgetbv",
            in("ecx") 0u32,
            out("eax") xcr0_lo,
            out("edx") xcr0_hi,
            options(nomem, nostack, preserves_flags),
        );
        let xcr0_lo = xcr0_lo | 0b1110_0111; // X87 | SSE | AVX | opmask | ZMM state
        core::arch::asm!(
            "xsetbv",
            in("ecx") 0u32,
            in("eax") xcr0_lo,
            in("edx") xcr0_hi,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// Issues a clean hardware shutdown or hypervisor exit via standard x86 I/O ports.
pub unsafe fn hardware_shutdown() -> ! {
    // 1. QEMU debug exit port (if enabled)
    crate::serial::outb(0x501, 0x00);
    crate::serial::outb(0xF4, 0x00);

    // 2. ACPI S5 Soft-Off Power Management I/O ports:
    // PIIX4 (QEMU default i440fx): PM1a_CNT register at 0xB004, SLP_TYP=0 | SLP_EN=1 (0x2000)
    crate::serial::outw(0xB004, 0x2000);
    // Q35 / microvm: PM1a_CNT register at 0x604, 0x2000
    crate::serial::outw(0x604, 0x2000);

    // 3. Fallback: PS/2 Keyboard Controller pulse reset
    crate::serial::outb(0x64, 0xFE);

    // 4. Halt CPU if still alive
    loop {
        core::arch::asm!("cli; hlt", options(nomem, nostack));
    }
}

#[cfg(test)]
mod memory_encryption_tests {
    use super::*;

    #[test]
    fn intel_tme_requires_active_locked_known_algorithm_for_strict_policy() {
        let active = decode_intel_tme((1 << 48) | (1 << 1) | 1);
        assert_eq!(
            active,
            MemoryEncryptionStatus::IntelTme {
                active: true,
                locked: true,
                algorithm: 1,
            }
        );
        assert!(active.satisfies_strict_policy());
        assert!(!decode_intel_tme(1 << 1).satisfies_strict_policy());
        assert!(!decode_intel_tme((3 << 48) | (1 << 1) | 1).satisfies_strict_policy());
    }

    #[test]
    fn amd_sme_decodes_status_and_validates_c_bit_position() {
        let active = decode_amd_sme(true, true, AMD_SYSCFG_MEM_ENCRYPTION_ENABLED, 47);
        assert_eq!(active.amd_c_bit_mask(), Some(1 << 47));
        assert!(active.satisfies_strict_policy());
        assert!(!decode_amd_sme(true, true, 0, 47).satisfies_strict_policy());
        assert!(
            !decode_amd_sme(true, false, AMD_SYSCFG_MEM_ENCRYPTION_ENABLED, 63)
                .satisfies_strict_policy()
        );
        assert_eq!(
            decode_amd_sme(false, true, AMD_SYSCFG_MEM_ENCRYPTION_ENABLED, 47).amd_c_bit_mask(),
            None
        );
    }

    #[test]
    fn unsupported_hardware_is_not_strictly_accepted() {
        assert!(!MemoryEncryptionStatus::Unsupported.satisfies_strict_policy());
        assert_eq!(MemoryEncryptionStatus::Unsupported.amd_c_bit_mask(), None);
    }
}
