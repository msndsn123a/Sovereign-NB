//! Intel L3 Cache Allocation Technology (CAT) setup for bare-metal inference.
//!
//! CAT MSRs are touched only after `boot::l3_cat_capabilities` reports support.

use crate::boot::L3CatCapabilities;

const IA32_PQR_ASSOC: u32 = 0xC8F;
const IA32_L3_QOS_CFG: u32 = 0xC81;
const IA32_L3_MASK_BASE: u32 = 0xC90;
const CLOS_INFERENCE: u32 = 1;
const L3_CDP_ENABLE: u64 = 1;

#[derive(Clone, Copy, Debug)]
pub struct L3CatConfiguration {
    pub way_mask: u32,
    pub qos_config: u64,
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

#[inline]
unsafe fn write_msr(msr: u32, value: u64) {
    core::arch::asm!(
        "wrmsr",
        in("ecx") msr,
        in("eax") value as u32,
        in("edx") (value >> 32) as u32,
        options(nomem, nostack, preserves_flags),
    );
}

/// Reserve the high-order cache ways for CLOS 1 and associate the current core.
///
/// Returns `None` if the CPU reports code/data prioritization (which changes the
/// mask-register layout), the resource geometry is unsupported, or readback
/// does not confirm the configuration. The caller must first CPUID-gate this
/// function; CAT support is the architectural guard for these MSR accesses.
///
/// # Safety
/// Must run at CPL0 on the CPU whose CPUID capabilities were probed. CAT MSR
/// access is valid only after `boot::l3_cat_capabilities` returns `Some`.
pub unsafe fn configure_l3_cat(capabilities: L3CatCapabilities) -> Option<L3CatConfiguration> {
    if capabilities.clos_count <= CLOS_INFERENCE || !(2..=32).contains(&capabilities.cbm_length) {
        return None;
    }

    let qos_config = read_msr(IA32_L3_QOS_CFG);
    if qos_config & L3_CDP_ENABLE != 0 {
        // Avoid writing an ambiguous code/data mask when CDP is already enabled.
        return None;
    }

    let ways = capabilities.cbm_length as u32;
    let high_way_count = ways / 2;
    let low_mask = (1u32 << high_way_count) - 1;
    let way_mask = low_mask << (ways - high_way_count);
    let mask_msr = IA32_L3_MASK_BASE + CLOS_INFERENCE;

    write_msr(mask_msr, way_mask as u64);
    let configured_mask = read_msr(mask_msr) as u32;
    if configured_mask != way_mask {
        return None;
    }

    let association = read_msr(IA32_PQR_ASSOC);
    let clos_and_rmid = ((CLOS_INFERENCE as u64) << 32) | (association & 0xFFFF_FFFF);
    write_msr(IA32_PQR_ASSOC, clos_and_rmid);
    let configured_association = read_msr(IA32_PQR_ASSOC);
    if (configured_association >> 32) as u32 != CLOS_INFERENCE {
        return None;
    }

    Some(L3CatConfiguration {
        way_mask,
        qos_config,
    })
}
