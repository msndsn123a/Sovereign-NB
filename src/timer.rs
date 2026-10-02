//! Calibrated hardware timestamping and fixed-storage latency auditing.

use core::sync::atomic::{AtomicU64, Ordering};

static TSC_FREQUENCY_HZ: AtomicU64 = AtomicU64::new(0);
static CALIBRATION_SOURCE: AtomicU64 = AtomicU64::new(0);
static TSC_RDTSCP_SUPPORTED: AtomicU64 = AtomicU64::new(0);

pub const MAX_LATENCY_SAMPLES: usize = 128;

#[derive(Clone, Copy, Debug, Default)]
pub struct LatencySample {
    pub ingress_cycles: u64,
    pub compute_cycles: u64,
    pub egress_cycles: u64,
    pub turnaround_cycles: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LatencySummary {
    pub min: u64,
    pub p50: u64,
    pub p99: u64,
    pub max: u64,
}

/// Reads TSC with a fence before and after RDTSC to serialize surrounding work.
#[inline(always)]
pub unsafe fn read_tsc() -> u64 {
    core::arch::x86_64::_mm_lfence();
    let value = core::arch::x86_64::_rdtsc();
    core::arch::x86_64::_mm_lfence();
    value
}

/// Reads a serialized TSC value and the RDTSCP processor auxiliary value.
/// The auxiliary value is zero when the processor does not support RDTSCP.
#[inline(always)]
pub unsafe fn read_tsc_with_aux() -> (u64, u32) {
    core::arch::x86_64::_mm_lfence();
    if TSC_RDTSCP_SUPPORTED.load(Ordering::Relaxed) != 0 {
        let low: u32;
        let high: u32;
        let processor_id: u32;
        core::arch::asm!(
            "rdtscp",
            "lfence",
            out("eax") low,
            out("edx") high,
            out("ecx") processor_id,
            options(nomem, nostack, preserves_flags)
        );
        (((high as u64) << 32) | low as u64, processor_id)
    } else {
        let cycles = core::arch::x86_64::_rdtsc();
        core::arch::x86_64::_mm_lfence();
        (cycles, 0)
    }
}

/// Calibrates from CPUID leaf 0x15 when it supplies a complete ratio, otherwise
/// measures a 100 ms Boot Services stall. Returns the measured frequency in Hz.
pub fn calibrate_tsc<F>(mut stall: F) -> u64
where
    F: FnMut(usize),
{
    TSC_RDTSCP_SUPPORTED.store(u64::from(cpuid_supports_rdtscp()), Ordering::Release);
    let cpuid_hz = cpuid_tsc_frequency_hz();
    let frequency_hz = if cpuid_hz != 0 {
        CALIBRATION_SOURCE.store(1, Ordering::Release);
        cpuid_hz
    } else {
        CALIBRATION_SOURCE.store(2, Ordering::Release);
        let start = unsafe { read_tsc() };
        stall(100_000);
        let elapsed = unsafe { read_tsc() }.saturating_sub(start);
        elapsed.saturating_mul(10)
    };

    TSC_FREQUENCY_HZ.store(frequency_hz, Ordering::Release);
    frequency_hz
}

fn cpuid_supports_rdtscp() -> bool {
    let max_extended_leaf = core::arch::x86_64::__cpuid(0x8000_0000).eax;
    max_extended_leaf >= 0x8000_0001
        && (core::arch::x86_64::__cpuid(0x8000_0001).edx & (1 << 27)) != 0
}

fn cpuid_tsc_frequency_hz() -> u64 {
    let max_leaf = core::arch::x86_64::__cpuid(0).eax;
    if max_leaf < 0x15 {
        return 0;
    }

    let leaf = core::arch::x86_64::__cpuid(0x15);
    if leaf.eax == 0 || leaf.ebx == 0 || leaf.ecx == 0 {
        return 0;
    }

    (leaf.ecx as u64).saturating_mul(leaf.ebx as u64) / leaf.eax as u64
}

pub fn tsc_frequency_hz() -> u64 {
    TSC_FREQUENCY_HZ.load(Ordering::Acquire)
}

pub fn tsc_frequency_mhz_parts() -> (u64, u64) {
    let frequency_hz = tsc_frequency_hz();
    let whole = frequency_hz / 1_000_000;
    let hundredths = (frequency_hz % 1_000_000) / 10_000;
    (whole, hundredths)
}

/// Returns thousandths of a microsecond using the calibrated TSC frequency.
pub fn cycles_to_micros_milli(cycles: u64) -> u64 {
    let frequency_hz = tsc_frequency_hz();
    if frequency_hz == 0 {
        return 0;
    }
    ((cycles as u128 * 1_000_000_000u128) / frequency_hz as u128).min(u64::MAX as u128) as u64
}

pub fn cycles_for_seconds(seconds: u64) -> u64 {
    tsc_frequency_hz().saturating_mul(seconds)
}

/// Summarizes a bounded sample slice in-place on the stack, without allocating.
pub fn summarize(samples: &[u64]) -> LatencySummary {
    if samples.is_empty() {
        return LatencySummary::default();
    }

    let mut sorted = [0u64; MAX_LATENCY_SAMPLES];
    let count = samples.len().min(MAX_LATENCY_SAMPLES);
    sorted[..count].copy_from_slice(&samples[..count]);

    let mut index = 1usize;
    while index < count {
        let value = sorted[index];
        let mut cursor = index;
        while cursor > 0 && sorted[cursor - 1] > value {
            sorted[cursor] = sorted[cursor - 1];
            cursor -= 1;
        }
        sorted[cursor] = value;
        index += 1;
    }

    let p50 = if count & 1 == 1 {
        sorted[count / 2]
    } else {
        let lower = sorted[count / 2 - 1];
        let upper = sorted[count / 2];
        lower / 2 + upper / 2 + ((lower & 1) + (upper & 1)) / 2
    };
    let p99_index = ((count * 99).div_ceil(100)).saturating_sub(1);

    LatencySummary {
        min: sorted[0],
        p50,
        p99: sorted[p99_index],
        max: sorted[count - 1],
    }
}

pub fn calibration_used_cpuid15() -> bool {
    CALIBRATION_SOURCE.load(Ordering::Acquire) == 1
}
