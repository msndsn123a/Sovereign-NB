//! Fixed-point activation lookup tables for inference.

pub use crate::wasm_math::{AlignedActivationLut, SILU_LUT_Q4};

const LUT_ENTRIES: usize = 256;

/// Q4-input/Q4-output SiLU approximation, indexed from signed input -128..127.
/// The aligned 512-byte table occupies exactly eight 64-byte cache lines.
/// One data-dependent table access and no runtime branch or floating point.
#[inline(always)]
pub fn silu_q4(input: i8) -> i16 {
    crate::wasm_math::silu_q4(input)
}

/// Touch every cache line at startup so a selected LUT path begins L1-warm.
/// Cache residency is hardware-managed and may later be displaced by other data.
#[inline(never)]
pub fn warm_silu_lut() -> i32 {
    let mut checksum = 0i32;
    let mut index = 0usize;
    while index < LUT_ENTRIES {
        let value = unsafe { core::ptr::read_volatile(&SILU_LUT_Q4.0[index]) };
        checksum = checksum.wrapping_add(value as i32);
        index += 32;
    }
    checksum
}
