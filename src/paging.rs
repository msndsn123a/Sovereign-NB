//! Static identity page tables with 1 GiB / 2 MiB huge-page mappings.
//!
//! The hierarchy maps the first 512 GiB of physical memory at identical virtual
//! addresses. UEFI supplies the current identity mapping until ExitBootServices;
//! `activate_identity_map` installs these tables only after that point.

use core::arch::x86_64::__cpuid;

const ENTRIES_PER_TABLE: usize = 512;
const PAGE_PRESENT: u64 = 1 << 0;
const PAGE_WRITABLE: u64 = 1 << 1;
const PAGE_CACHE_DISABLE: u64 = 1 << 4;
const PAGE_HUGE: u64 = 1 << 7;
const PAGE_2M_SIZE: u64 = 2 * 1024 * 1024;
const PAGE_1G_SIZE: u64 = 1024 * 1024 * 1024;
const ADDRESS_SPACE_BYTES: u64 = PAGE_1G_SIZE * ENTRIES_PER_TABLE as u64;
const CR4_PGE: u64 = 1 << 7;

#[repr(C, align(4096))]
struct PageTable([u64; ENTRIES_PER_TABLE]);

impl PageTable {
    const fn zeroed() -> Self {
        Self([0; ENTRIES_PER_TABLE])
    }
}

#[repr(C, align(4096))]
struct PageDirectory([PageTable; ENTRIES_PER_TABLE]);

static mut PML4: PageTable = PageTable::zeroed();
static mut PDPT: PageTable = PageTable::zeroed();
static mut PAGE_DIRECTORY: PageDirectory =
    PageDirectory([const { PageTable::zeroed() }; ENTRIES_PER_TABLE]);

#[derive(Clone, Copy, Debug, Default)]
pub struct MmioRange {
    pub base: u64,
    pub length: u64,
}

impl MmioRange {
    pub const EMPTY: Self = Self { base: 0, length: 0 };

    pub const fn new(base: u64, length: u64) -> Self {
        Self { base, length }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HugePageMode {
    OneGiB,
    TwoMiB,
}

#[derive(Clone, Copy, Debug)]
pub struct PagingActivation {
    pub mode: HugePageMode,
    pub mapped_bytes: u64,
    pub uncached_huge_pages: usize,
    pub encrypted_huge_pages: usize,
    pub encryption_mask: u64,
}

pub fn supports_1g_pages() -> bool {
    // CPUID.80000000:EAX must enumerate leaf 80000001h.
    if __cpuid(0x8000_0000).eax < 0x8000_0001 {
        return false;
    }
    (__cpuid(0x8000_0001).edx & (1 << 26)) != 0
}

fn intersects_mmio(page_base: u64, page_size: u64, ranges: &[MmioRange]) -> bool {
    let page_end = page_base.saturating_add(page_size);
    ranges.iter().any(|range| {
        if range.length == 0 {
            return false;
        }
        let range_end = range.base.saturating_add(range.length);
        range.base < page_end && page_base < range_end
    })
}

fn huge_page_entry(
    physical: u64,
    page_size: u64,
    mmio_ranges: &[MmioRange],
    encryption_mask: u64,
) -> (u64, bool, bool) {
    let mut entry = physical | PAGE_PRESENT | PAGE_WRITABLE | PAGE_HUGE;
    if intersects_mmio(physical, page_size, mmio_ranges) {
        entry |= PAGE_CACHE_DISABLE;
        (entry, true, false)
    } else if encryption_mask != 0 {
        (entry | encryption_mask, false, true)
    } else {
        (entry, false, false)
    }
}

/// Installs the static identity map and switches CR3.
///
/// # Safety
/// Call only after ExitBootServices, while executing with interrupts disabled.
/// The fixed mapping covers physical/virtual addresses below 512 GiB; every
/// active code, stack, DMA and device address must remain inside that window.
pub unsafe fn activate_identity_map(
    mmio_ranges: &[MmioRange],
    encryption_mask: Option<u64>,
) -> PagingActivation {
    let pml4 = core::ptr::addr_of_mut!(PML4.0).cast::<u64>();
    let pdpt = core::ptr::addr_of_mut!(PDPT.0).cast::<u64>();
    let directories = core::ptr::addr_of_mut!(PAGE_DIRECTORY.0).cast::<PageTable>();
    let encryption_mask = encryption_mask.unwrap_or(0);
    let pml4_address = (pml4 as u64) | encryption_mask;
    let pdpt_address = (pdpt as u64) | encryption_mask;
    core::ptr::write_bytes(pml4, 0, ENTRIES_PER_TABLE);
    core::ptr::write_bytes(pdpt, 0, ENTRIES_PER_TABLE);
    pml4.write(pdpt_address | PAGE_PRESENT | PAGE_WRITABLE);

    let mut encrypted_huge_pages = 0usize;
    let (mode, uncached_huge_pages) = if supports_1g_pages() {
        let mut uncached = 0usize;
        let mut index = 0usize;
        while index < ENTRIES_PER_TABLE {
            let physical = index as u64 * PAGE_1G_SIZE;
            let (entry, is_uncached, is_encrypted) =
                huge_page_entry(physical, PAGE_1G_SIZE, mmio_ranges, encryption_mask);
            if is_uncached {
                uncached += 1;
            }
            if is_encrypted {
                encrypted_huge_pages += 1;
            }
            pdpt.add(index).write(entry);
            index += 1;
        }
        (HugePageMode::OneGiB, uncached)
    } else {
        let mut uncached = 0usize;
        let mut gigabyte_index = 0usize;
        while gigabyte_index < ENTRIES_PER_TABLE {
            let directory = directories.add(gigabyte_index).cast::<u64>();
            let directory_address = (directories.add(gigabyte_index) as u64) | encryption_mask;
            pdpt.add(gigabyte_index)
                .write(directory_address | PAGE_PRESENT | PAGE_WRITABLE);

            let gigabyte_base = gigabyte_index as u64 * PAGE_1G_SIZE;
            let mut page_index = 0usize;
            while page_index < ENTRIES_PER_TABLE {
                let physical = gigabyte_base + page_index as u64 * PAGE_2M_SIZE;
                let (entry, is_uncached, is_encrypted) =
                    huge_page_entry(physical, PAGE_2M_SIZE, mmio_ranges, encryption_mask);
                if is_uncached {
                    uncached += 1;
                }
                if is_encrypted {
                    encrypted_huge_pages += 1;
                }
                directory.add(page_index).write(entry);
                page_index += 1;
            }
            gigabyte_index += 1;
        }
        (HugePageMode::TwoMiB, uncached)
    };

    // Switch CR3 after UEFI services have ended. Toggle PGE to invalidate any
    // global translations left by firmware, then restore the original CR4 value.
    let old_cr4: u64;
    core::arch::asm!("mov {}, cr4", out(reg) old_cr4, options(nomem, nostack, preserves_flags));
    if old_cr4 & CR4_PGE != 0 {
        let without_pge = old_cr4 & !CR4_PGE;
        core::arch::asm!("mov cr4, {}", in(reg) without_pge, options(nostack, preserves_flags));
    }
    core::arch::asm!("mov cr3, {}", in(reg) pml4_address, options(nostack, preserves_flags));
    if old_cr4 & CR4_PGE != 0 {
        core::arch::asm!("mov cr4, {}", in(reg) old_cr4, options(nostack, preserves_flags));
    }

    PagingActivation {
        mode,
        mapped_bytes: ADDRESS_SPACE_BYTES,
        uncached_huge_pages,
        encrypted_huge_pages,
        encryption_mask,
    }
}

const _: () = assert!(core::mem::align_of::<PageTable>() == 4096);
const _: () = assert!(core::mem::size_of::<PageTable>() == 4096);
const _: () = assert!(core::mem::align_of::<PageDirectory>() == 4096);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encryption_c_bit_tags_ram_entries_but_never_mmio_entries() {
        let mask = 1u64 << 47;
        let (ram_entry, ram_uncached, ram_encrypted) = huge_page_entry(0, PAGE_2M_SIZE, &[], mask);
        assert!(!ram_uncached);
        assert!(ram_encrypted);
        assert_ne!(ram_entry & mask, 0);

        let mmio = [MmioRange::new(0x20_0000, 4096)];
        let (mmio_entry, mmio_uncached, mmio_encrypted) =
            huge_page_entry(0x20_0000, PAGE_2M_SIZE, &mmio, mask);
        assert!(mmio_uncached);
        assert!(!mmio_encrypted);
        assert_eq!(mmio_entry & mask, 0);
    }
}
