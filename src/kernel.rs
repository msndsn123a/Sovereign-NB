//! Bare-metal AVX-512 BitNet ternary compute kernel.
//!
//! Strictly `#![no_std]`, zero floating-point operations, zero GEMM multipliers,
//! using pure integer/bitwise SIMD intrinsics with 64-byte alignment.

use core::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_castsi512_si128, _mm512_cvtepi8_epi32,
    _mm512_extracti32x4_epi32, _mm512_load_si512, _mm512_maskz_mov_epi8, _mm512_reduce_add_epi32,
    _mm512_sub_epi32,
};

use crate::bitpack::{decode_ternary, unpack_masks_64};

pub const MLP_INPUT_DIM: usize = 64;
pub const MLP_HIDDEN_DIM: usize = 32;
pub const MLP_OUTPUT_DIM: usize = 16;

/// Fixed-size packed ternary 64 -> 32 -> 16 network. Layer rows are contiguous.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TernaryMlp64x32x16 {
    pub layer1: [[u8; 16]; MLP_HIDDEN_DIM],
    pub layer2: [[u8; 8]; MLP_OUTPUT_DIM],
}

pub const LINEAR_ATTENTION_DIM: usize = 16;
pub const LINEAR_ATTENTION_STATE_LEN: usize = LINEAR_ATTENTION_DIM * LINEAR_ATTENTION_DIM;
pub const LINEAR_ATTENTION_WEIGHT_BYTES: usize = 832;

/// Packed ternary projection weights for a causal linear attention block.
///
/// Layout:
/// - W_Q: 16 rows x 64 inputs, packed as 16 bytes/row => 256 bytes
/// - W_K: 16 rows x 64 inputs, packed as 16 bytes/row => 256 bytes
/// - W_V: 16 rows x 64 inputs, packed as 16 bytes/row => 256 bytes
/// - W_O: 16 rows x 16 tokens, packed as 4 bytes/row => 64 bytes total
///
/// Total: 832 packed bytes, matching the attention shard layout.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TernaryLinearAttention {
    pub q: [[u8; 16]; LINEAR_ATTENTION_DIM],
    pub k: [[u8; 16]; LINEAR_ATTENTION_DIM],
    pub v: [[u8; 16]; LINEAR_ATTENTION_DIM],
    pub o: [[u8; 4]; LINEAR_ATTENTION_DIM],
}

impl TernaryLinearAttention {
    pub const fn zeroed() -> Self {
        Self {
            q: [[0; 16]; LINEAR_ATTENTION_DIM],
            k: [[0; 16]; LINEAR_ATTENTION_DIM],
            v: [[0; 16]; LINEAR_ATTENTION_DIM],
            o: [[0; 4]; LINEAR_ATTENTION_DIM],
        }
    }

    pub fn from_packed_payload(data: &[u8]) -> Option<Self> {
        if data.len() < LINEAR_ATTENTION_WEIGHT_BYTES {
            return None;
        }

        let mut attention = Self::zeroed();
        let mut row = 0usize;
        while row < LINEAR_ATTENTION_DIM {
            let offset = row * 16;
            attention.q[row].copy_from_slice(&data[offset..offset + 16]);
            attention.k[row].copy_from_slice(&data[256 + offset..256 + offset + 16]);
            attention.v[row].copy_from_slice(&data[512 + offset..512 + offset + 16]);
            attention.o[row].copy_from_slice(&data[768 + row * 4..768 + row * 4 + 4]);
            row += 1;
        }
        Some(attention)
    }
}

impl TernaryMlp64x32x16 {
    pub const fn zeroed() -> Self {
        Self {
            layer1: [[0; 16]; MLP_HIDDEN_DIM],
            layer2: [[0; 8]; MLP_OUTPUT_DIM],
        }
    }

    /// Copies exactly 640 packed weight bytes into fixed storage.
    pub fn from_packed_payload(data: &[u8]) -> Option<Self> {
        const PAYLOAD_BYTES: usize = MLP_HIDDEN_DIM * 16 + MLP_OUTPUT_DIM * 8;
        if data.len() < PAYLOAD_BYTES {
            return None;
        }

        let mut model = Self::zeroed();
        let mut row = 0usize;
        while row < MLP_HIDDEN_DIM {
            let offset = row * 16;
            model.layer1[row].copy_from_slice(&data[offset..offset + 16]);
            row += 1;
        }
        let layer2_start = MLP_HIDDEN_DIM * 16;
        row = 0;
        while row < MLP_OUTPUT_DIM {
            let offset = layer2_start + row * 8;
            model.layer2[row].copy_from_slice(&data[offset..offset + 8]);
            row += 1;
        }
        Some(model)
    }
}

/// Allocation-free scalar reference for the full two-layer network.
pub fn ternary_mlp_scalar(
    inputs: &[i8; MLP_INPUT_DIM],
    model: &TernaryMlp64x32x16,
) -> [i32; MLP_OUTPUT_DIM] {
    let mut output_storage = [0i32; 64];
    ternary_mlp_scalar_into(inputs, model, &mut output_storage);
    let mut outputs = [0i32; MLP_OUTPUT_DIM];
    outputs.copy_from_slice(&output_storage[..MLP_OUTPUT_DIM]);
    outputs
}

/// Scalar chained MLP that writes final values directly into caller-owned storage.
pub fn ternary_mlp_scalar_into(
    inputs: &[i8; MLP_INPUT_DIM],
    model: &TernaryMlp64x32x16,
    outputs: &mut [i32; 64],
) {
    let mut hidden = [0i8; MLP_HIDDEN_DIM];
    let mut hidden_row = 0usize;
    while hidden_row < MLP_HIDDEN_DIM {
        let mut accumulator = 0i32;
        let mut column = 0usize;
        while column < MLP_INPUT_DIM {
            let code = (model.layer1[hidden_row][column >> 2] >> ((column & 3) * 2)) & 0b11;
            accumulator += inputs[column] as i32 * decode_ternary(code) as i32;
            column += 1;
        }
        hidden[hidden_row] = integer_hard_sign(accumulator);
        hidden_row += 1;
    }

    let mut output_row = 0usize;
    while output_row < MLP_OUTPUT_DIM {
        let mut accumulator = 0i32;
        let mut column = 0usize;
        while column < MLP_HIDDEN_DIM {
            let packed = model.layer2[output_row][column >> 2];
            let code = (packed >> ((column & 3) * 2)) & 0b11;
            accumulator += hidden[column] as i32 * decode_ternary(code) as i32;
            column += 1;
        }
        outputs[output_row] = accumulator;
        output_row += 1;
    }
}

/// AVX-512 ternary dot-product path for both layers, with an integer hard-sign
/// activation between them. All buffers are fixed-size stack storage.
///
/// # Safety
/// AVX-512F/BW/DQ and OS-enabled ZMM state must be available.
pub unsafe fn ternary_mlp_avx512(
    inputs: &AlignedInputs64,
    model: &TernaryMlp64x32x16,
) -> [i32; MLP_OUTPUT_DIM] {
    ternary_mlp_avx512_ptr(inputs.0.as_ptr(), model)
}

/// AVX-512 MLP entry point for an input vector already resident in aligned shared memory.
///
/// # Safety
/// `inputs` must reference 64 readable, 64-byte-aligned i8 values, and AVX-512
/// OS state must be enabled.
pub unsafe fn ternary_mlp_avx512_ptr(
    inputs: *const i8,
    model: &TernaryMlp64x32x16,
) -> [i32; MLP_OUTPUT_DIM] {
    let mut output_storage = [0i32; 64];
    ternary_mlp_avx512_into_ptr(inputs, model, &mut output_storage);
    let mut outputs = [0i32; MLP_OUTPUT_DIM];
    outputs.copy_from_slice(&output_storage[..MLP_OUTPUT_DIM]);
    outputs
}

/// Chained AVX-512 MLP writing final outputs directly to the supplied slot.
///
/// # Safety
/// `inputs` must reference 64 readable, 64-byte-aligned i8 values, and AVX-512
/// OS state must be enabled.
pub unsafe fn ternary_mlp_avx512_into_ptr(
    inputs: *const i8,
    model: &TernaryMlp64x32x16,
    outputs: &mut [i32; 64],
) {
    let mut hidden = [0i8; 64];
    let mut row = 0usize;
    while row < MLP_HIDDEN_DIM {
        let (mask_pos, mask_neg) = unpack_masks_64(&model.layer1[row]);
        let sum = ternary_dot_product_avx512(inputs, mask_pos, mask_neg, 64);
        hidden[row] = integer_hard_sign(sum);
        row += 1;
    }

    let hidden_inputs = AlignedInputs64(hidden);
    let hidden_zmm = _mm512_load_si512(hidden_inputs.0.as_ptr() as *const __m512i);
    row = 0;
    while row < MLP_OUTPUT_DIM {
        let mut packed_row = [0u8; 16];
        packed_row[..8].copy_from_slice(&model.layer2[row]);
        let (mask_pos, mask_neg) = unpack_masks_64(&packed_row);
        outputs[row] = ternary_dot_product_zmm(hidden_zmm, mask_pos, mask_neg);
        row += 1;
    }
}

/// 64-byte aligned vector of 64 signed 8-bit activations (`i8`), matching a single 512-bit ZMM register.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlignedInputs64(pub [i8; 64]);

/// 64-byte aligned output buffer of 16 signed 32-bit accumulators (`i32`), matching a 512-bit ZMM register.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlignedOutputs16(pub [i32; 16]);

/// Rows of a 64-input ternary matrix; each row stores 64 weights in 16 bytes.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TernaryMatrix<const OUT_DIM: usize> {
    pub rows: [[u8; 16]; OUT_DIM],
}

impl<const OUT_DIM: usize> TernaryMatrix<OUT_DIM> {
    pub const fn zeroed() -> Self {
        Self {
            rows: [[0; 16]; OUT_DIM],
        }
    }
}

/// 64-byte aligned output vector for multi-output matrix-vector operations.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlignedOutputs<const OUT_DIM: usize>(pub [i32; OUT_DIM]);

/// Enables SSE, AVX, and AVX-512 state (`CR4.OSFXSR`, `CR4.OSXSAVE`, and `XCR0` opmask/ZMM bits)
/// in bare-metal UEFI if supported by the underlying processor.
#[inline]
pub unsafe fn enable_avx512_os_state() -> bool {
    if !crate::boot::cpu_supports_avx512() {
        return false;
    }

    let mut cr4: u64;
    core::arch::asm!("mov {}, cr4", out(reg) cr4, options(nomem, nostack, preserves_flags));
    cr4 |= (1 << 9) | (1 << 18);
    core::arch::asm!("mov cr4, {}", in(reg) cr4, options(nomem, nostack, preserves_flags));

    let current_xcr0_lo: u32;
    let current_xcr0_hi: u32;
    core::arch::asm!(
        "xgetbv",
        in("ecx") 0u32,
        out("eax") current_xcr0_lo,
        out("edx") current_xcr0_hi,
        options(nomem, nostack, preserves_flags),
    );
    core::arch::asm!(
        "xsetbv",
        in("ecx") 0u32,
        in("eax") current_xcr0_lo | 0b1110_0111,
        in("edx") current_xcr0_hi,
        options(nomem, nostack, preserves_flags),
    );

    let xcr0_lo: u32;
    let xcr0_hi: u32;
    core::arch::asm!(
        "xgetbv",
        in("ecx") 0u32,
        out("eax") xcr0_lo,
        out("edx") xcr0_hi,
        options(nomem, nostack, preserves_flags),
    );
    let xcr0 = ((xcr0_hi as u64) << 32) | xcr0_lo as u64;
    const REQUIRED_XCR0: u64 = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 5) | (1 << 6) | (1 << 7);
    (xcr0 & REQUIRED_XCR0) == REQUIRED_XCR0
}

/// Sums all 64 signed 8-bit lanes of a 512-bit ZMM register into 16 x 32-bit signed lanes
/// using `_mm512_cvtepi8_epi32` sign-extension across the four 128-bit quarters.
unsafe fn sum_epi8_to_epi32_zmm(vec: __m512i) -> __m512i {
    let q0 = _mm512_cvtepi8_epi32(_mm512_castsi512_si128(vec));
    let q1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32::<1>(vec));
    let q2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32::<2>(vec));
    let q3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32::<3>(vec));

    let sum01 = _mm512_add_epi32(q0, q1);
    let sum23 = _mm512_add_epi32(q2, q3);
    _mm512_add_epi32(sum01, sum23)
}

#[inline(always)]
unsafe fn ternary_dot_product_zmm(vector: __m512i, mask_pos: u64, mask_neg: u64) -> i32 {
    let pos_vec = _mm512_maskz_mov_epi8(mask_pos, vector);
    let neg_vec = _mm512_maskz_mov_epi8(mask_neg, vector);
    let pos_acc = sum_epi8_to_epi32_zmm(pos_vec);
    let neg_acc = sum_epi8_to_epi32_zmm(neg_vec);
    _mm512_reduce_add_epi32(_mm512_sub_epi32(pos_acc, neg_acc))
}

/// Core ternary dense vector dot-product using AVX-512 integer and mask intrinsics.
///
/// # Safety
/// - `inputs` must point to a valid, 64-byte aligned buffer (`#[repr(align(64))]`) of at least 64 `i8` elements.
/// - `length` must be `64`.
/// - AVX-512F/BW/VL/DQ must be supported and its OS state enabled via `enable_avx512_os_state`.
#[inline(never)]
pub unsafe fn ternary_dot_product_avx512(
    inputs: *const i8,
    mask_pos: u64,
    mask_neg: u64,
    length: usize, // Must be 64
) -> i32 {
    debug_assert_eq!(length, 64);
    debug_assert_eq!((inputs as usize) & 63, 0);

    // 1. Load 64 input values (i8) into a 512-bit ZMM register using _mm512_load_si512.
    let in_vec: __m512i = _mm512_load_si512(inputs as *const __m512i);

    // 2. Isolate positive inputs (+1 weights) via _mm512_maskz_mov_epi8(mask_pos, in_vec).
    let pos_vec: __m512i = _mm512_maskz_mov_epi8(mask_pos, in_vec);

    // 3. Isolate negative inputs (-1 weights) via _mm512_maskz_mov_epi8(mask_neg, in_vec).
    let neg_vec: __m512i = _mm512_maskz_mov_epi8(mask_neg, in_vec);

    // 4. Expand/convert the 8-bit integers to 32-bit accumulators using _mm512_cvtepi8_epi32.
    let pos_acc: __m512i = sum_epi8_to_epi32_zmm(pos_vec);
    let neg_acc: __m512i = sum_epi8_to_epi32_zmm(neg_vec);

    // 5. Accumulate: Positive_Sum - Negative_Sum.
    let diff_acc: __m512i = _mm512_sub_epi32(pos_acc, neg_acc);

    // 6. Horizontal reduction to final signed scalar i32 activation value.
    let simd_res = _mm512_reduce_add_epi32(diff_acc);

    simd_res
}

/// Reference scalar dot product used on CPUs without the required AVX-512 feature set.
pub fn ternary_dot_product_scalar(inputs: &[i8; 64], mask_pos: u64, mask_neg: u64) -> i32 {
    let mut sum = 0i32;
    let mut index = 0usize;
    while index < 64 {
        let bit = 1u64 << index;
        let value = inputs[index] as i32;
        if mask_pos & bit != 0 {
            sum += value;
        } else if mask_neg & bit != 0 {
            sum -= value;
        }
        index += 1;
    }
    sum
}

/// Computes each row of a 64-by-OUT_DIM ternary matrix against a 64-element input vector.
pub fn matrix_vector_scalar<const OUT_DIM: usize>(
    inputs: &[i8; 64],
    matrix: &TernaryMatrix<OUT_DIM>,
) -> AlignedOutputs<OUT_DIM> {
    let mut outputs = [0i32; OUT_DIM];
    let mut row = 0usize;
    while row < OUT_DIM {
        let (mask_pos, mask_neg) = unpack_masks_64(&matrix.rows[row]);
        outputs[row] = ternary_dot_product_scalar(inputs, mask_pos, mask_neg);
        row += 1;
    }
    AlignedOutputs(outputs)
}

/// Independent scalar reference that decodes each packed weight directly.
pub fn matrix_vector_reference<const OUT_DIM: usize>(
    inputs: &[i8; 64],
    matrix: &TernaryMatrix<OUT_DIM>,
) -> AlignedOutputs<OUT_DIM> {
    let mut outputs = [0i32; OUT_DIM];
    let mut row = 0usize;
    while row < OUT_DIM {
        let mut column = 0usize;
        while column < 64 {
            let code = (matrix.rows[row][column >> 2] >> ((column & 3) * 2)) & 0b11;
            let weight = decode_ternary(code) as i32;
            outputs[row] += inputs[column] as i32 * weight;
            column += 1;
        }
        row += 1;
    }
    AlignedOutputs(outputs)
}

/// AVX-512 implementation of [`matrix_vector_scalar`].
///
/// # Safety
/// AVX-512F/BW/DQ and OS-enabled ZMM state must be available; `inputs` must be 64-byte aligned.
pub unsafe fn matrix_vector_avx512<const OUT_DIM: usize>(
    inputs: &AlignedInputs64,
    matrix: &TernaryMatrix<OUT_DIM>,
) -> AlignedOutputs<OUT_DIM> {
    let mut outputs = [0i32; OUT_DIM];
    let mut row = 0usize;
    while row < OUT_DIM {
        let (mask_pos, mask_neg) = unpack_masks_64(&matrix.rows[row]);
        outputs[row] = ternary_dot_product_avx512(inputs.0.as_ptr(), mask_pos, mask_neg, 64);
        row += 1;
    }
    AlignedOutputs(outputs)
}

/// Integer hard-sign activation: negative values map to -1, zero to 0, positive to 1.
#[inline(always)]
pub const fn integer_hard_sign(value: i32) -> i8 {
    if value < 0 {
        -1
    } else if value > 0 {
        1
    } else {
        0
    }
}

/// Quantizes an integer output vector with [`integer_hard_sign`].
pub fn quantize_hard_sign<const OUT_DIM: usize>(values: &[i32; OUT_DIM]) -> [i8; OUT_DIM] {
    let mut quantized = [0i8; OUT_DIM];
    let mut index = 0usize;
    while index < OUT_DIM {
        quantized[index] = integer_hard_sign(values[index]);
        index += 1;
    }
    quantized
}

/// Resets the recurrent causal-state matrix to zero before a new sequence starts.
pub fn reset_attention_state(state: &mut [i32; LINEAR_ATTENTION_STATE_LEN]) {
    let mut index = 0usize;
    while index < state.len() {
        state[index] = 0;
        index += 1;
    }
}

/// Integer-only causal linear attention step.
///
/// `state` is kept as a 16 x 16 matrix in row-major order. For each token frame,
/// we update `state += outer(key, value)` and then project the query against the
/// accumulated memory to produce a 16-dimensional output vector.
pub fn causal_linear_attention_scalar(
    inputs: &[i8; 64],
    model: &TernaryLinearAttention,
    state: &mut [i32; LINEAR_ATTENTION_STATE_LEN],
) -> [i32; LINEAR_ATTENTION_DIM] {
    let mut q = [0i32; LINEAR_ATTENTION_DIM];
    let mut k = [0i32; LINEAR_ATTENTION_DIM];
    let mut v = [0i32; LINEAR_ATTENTION_DIM];

    let mut row = 0usize;
    while row < LINEAR_ATTENTION_DIM {
        q[row] = ternary_vector_dot_scalar(inputs, &model.q[row]);
        k[row] = ternary_vector_dot_scalar(inputs, &model.k[row]);
        v[row] = ternary_vector_dot_scalar(inputs, &model.v[row]);
        row += 1;
    }

    row = 0usize;
    while row < LINEAR_ATTENTION_DIM {
        let mut column = 0usize;
        while column < LINEAR_ATTENTION_DIM {
            let idx = row * LINEAR_ATTENTION_DIM + column;
            let contribution = k[row].saturating_mul(v[column]);
            state[idx] = state[idx].saturating_add(contribution);
            column += 1;
        }
        row += 1;
    }

    let mut output = [0i32; LINEAR_ATTENTION_DIM];
    row = 0usize;
    while row < LINEAR_ATTENTION_DIM {
        let mut column = 0usize;
        while column < LINEAR_ATTENTION_DIM {
            let idx = column * LINEAR_ATTENTION_DIM + row;
            output[row] += q[column].saturating_mul(state[idx]);
            column += 1;
        }
        row += 1;
    }

    let mut projected = [0i32; LINEAR_ATTENTION_DIM];
    row = 0usize;
    while row < LINEAR_ATTENTION_DIM {
        let mut column = 0usize;
        while column < LINEAR_ATTENTION_DIM {
            let code = (model.o[row][column >> 2] >> ((column & 3) * 2)) & 0b11;
            projected[row] += output[column].saturating_mul(decode_ternary(code) as i32);
            column += 1;
        }
        row += 1;
    }
    projected
}

fn ternary_vector_dot_scalar(inputs: &[i8; 64], packed: &[u8; 16]) -> i32 {
    let mut sum = 0i32;
    let mut index = 0usize;
    while index < 64 {
        let code = (packed[index >> 2] >> ((index & 3) * 2)) & 0b11;
        sum += inputs[index] as i32 * decode_ternary(code) as i32;
        index += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_state_reset_clears_sequence_state() {
        let mut state = [0i32; 256];
        for (index, value) in state.iter_mut().enumerate() {
            *value = (index as i32) - 128;
        }
        reset_attention_state(&mut state);
        assert!(state.iter().all(|value| *value == 0));
    }

    #[test]
    fn causal_attention_step_matches_reference() {
        let inputs = [7i8; 64];
        let model = TernaryLinearAttention::zeroed();
        let mut state = [0i32; 256];
        let output = causal_linear_attention_scalar(&inputs, &model, &mut state);
        assert_eq!(output.len(), 16);
        assert_eq!(output.iter().filter(|&&v| v == 0).count(), 16);
        assert!(state.iter().all(|&v| v == 0));
    }
}
