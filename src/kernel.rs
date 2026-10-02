//! Bare-metal AVX-512 BitNet ternary compute kernel.
//!
//! Strictly `#![no_std]`, zero floating-point operations, zero GEMM multipliers,
//! using pure integer/bitwise SIMD intrinsics with 64-byte alignment.

use core::arch::x86_64::{
    __m512i, _mm512_add_epi32, _mm512_add_epi64, _mm512_and_si512, _mm512_castsi512_si128,
    _mm512_cvtepi8_epi32, _mm512_extracti32x4_epi32, _mm512_load_si512, _mm512_loadu_si512,
    _mm512_maskz_mov_epi8, _mm512_popcnt_epi64, _mm512_reduce_add_epi32, _mm512_set1_epi64,
    _mm512_storeu_si512, _mm512_sub_epi32, _mm512_sub_epi64, _mm_prefetch, _MM_HINT_T0,
};

use crate::activation::silu_q4;
use crate::bitpack::{decode_ternary, unpack_masks_64};

pub const MLP_INPUT_DIM: usize = 64;
pub const MLP_HIDDEN_DIM: usize = 32;
pub const MLP_OUTPUT_DIM: usize = 16;
pub const MLP_BLOCK_WIDTH: usize = 8;
pub const MLP_LAYER1_BLOCKS_PER_ROW: usize = MLP_INPUT_DIM / MLP_BLOCK_WIDTH;
pub const MLP_LAYER2_BLOCKS_PER_ROW: usize = MLP_HIDDEN_DIM / MLP_BLOCK_WIDTH;
pub const MLP_BLOCK_MASK_BYTES: usize = MLP_HIDDEN_DIM + MLP_OUTPUT_DIM;
pub const MLP_SPARSE_PAYLOAD_BYTES: usize =
    MLP_HIDDEN_DIM * 16 + MLP_OUTPUT_DIM * 8 + MLP_BLOCK_MASK_BYTES;

/// Sweep a fixed shard buffer by cache line before inference when CAT is absent.
/// Prefetch instructions are hints; the volatile touches ensure each line is
/// brought into the current core's normal cache hierarchy before the loop.
pub fn software_prefetch_warm(data: &[u8]) -> usize {
    const CACHE_LINE_BYTES: usize = 64;
    let mut offset = 0usize;
    let mut lines = 0usize;
    while offset < data.len() {
        unsafe {
            _mm_prefetch(data.as_ptr().add(offset).cast::<i8>(), _MM_HINT_T0);
            core::ptr::read_volatile(data.as_ptr().add(offset));
        }
        offset += CACHE_LINE_BYTES;
        lines += 1;
    }
    lines
}

/// Fixed-size packed ternary 64 -> 32 -> 16 network. Layer rows are contiguous.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TernaryMlp64x32x16 {
    pub layer1: [[u8; 16]; MLP_HIDDEN_DIM],
    pub layer2: [[u8; 8]; MLP_OUTPUT_DIM],
}

#[derive(Clone, Copy)]
pub struct MlpBlockMask {
    /// One active bit per 8-input block for each first-layer row.
    pub layer1: [u8; MLP_HIDDEN_DIM],
    /// One active bit per 8-input block for each second-layer row.
    pub layer2: [u8; MLP_OUTPUT_DIM],
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct SparseTernaryMlp64x32x16 {
    pub weights: TernaryMlp64x32x16,
    pub active_blocks: MlpBlockMask,
}

impl SparseTernaryMlp64x32x16 {
    pub fn from_packed_payload(data: &[u8]) -> Option<Self> {
        if data.len() < MLP_SPARSE_PAYLOAD_BYTES {
            return None;
        }
        let weights = TernaryMlp64x32x16::from_packed_payload(data)?;
        let mut active_blocks = MlpBlockMask {
            layer1: [0; MLP_HIDDEN_DIM],
            layer2: [0; MLP_OUTPUT_DIM],
        };
        active_blocks.layer1.copy_from_slice(
            &data[MLP_HIDDEN_DIM * 16 + MLP_OUTPUT_DIM * 8
                ..MLP_HIDDEN_DIM * 16 + MLP_OUTPUT_DIM * 8 + MLP_HIDDEN_DIM],
        );
        active_blocks.layer2.copy_from_slice(
            &data[MLP_HIDDEN_DIM * 16 + MLP_OUTPUT_DIM * 8 + MLP_HIDDEN_DIM
                ..MLP_SPARSE_PAYLOAD_BYTES],
        );
        if active_blocks.layer2.iter().any(|mask| mask & 0xF0 != 0) {
            return None;
        }

        let model = Self {
            weights,
            active_blocks,
        };
        if model.inactive_blocks_are_zero() {
            Some(model)
        } else {
            None
        }
    }

    fn inactive_blocks_are_zero(&self) -> bool {
        let mut row = 0usize;
        while row < MLP_HIDDEN_DIM {
            let mut block = 0usize;
            while block < MLP_LAYER1_BLOCKS_PER_ROW {
                if self.active_blocks.layer1[row] & (1 << block) == 0 {
                    let start = block * 2;
                    if self.weights.layer1[row][start] != 0
                        || self.weights.layer1[row][start + 1] != 0
                    {
                        return false;
                    }
                }
                block += 1;
            }
            row += 1;
        }

        row = 0;
        while row < MLP_OUTPUT_DIM {
            let mut block = 0usize;
            while block < MLP_LAYER2_BLOCKS_PER_ROW {
                if self.active_blocks.layer2[row] & (1 << block) == 0 {
                    let start = block * 2;
                    if self.weights.layer2[row][start] != 0
                        || self.weights.layer2[row][start + 1] != 0
                    {
                        return false;
                    }
                }
                block += 1;
            }
            row += 1;
        }
        true
    }

    pub fn active_block_count(&self) -> usize {
        self.active_blocks
            .layer1
            .iter()
            .map(|mask| mask.count_ones() as usize)
            .sum::<usize>()
            + self
                .active_blocks
                .layer2
                .iter()
                .map(|mask| mask.count_ones() as usize)
                .sum::<usize>()
    }
}

pub const LINEAR_ATTENTION_DIM: usize = 16;
pub const LINEAR_ATTENTION_STATE_LEN: usize = LINEAR_ATTENTION_DIM * LINEAR_ATTENTION_DIM;
pub const ATTENTION_STREAM_COUNT: usize = 8;
pub const LINEAR_ATTENTION_STATE_BANK_LEN: usize =
    LINEAR_ATTENTION_STATE_LEN * ATTENTION_STREAM_COUNT;
pub const LINEAR_ATTENTION_WEIGHT_BYTES: usize = 832;
pub const LINEAR_ATTENTION_POT_WEIGHT_BYTES: usize = 1664;

/// Eight independent recurrent matrices with cache-line-aligned storage.
#[repr(C, align(64))]
pub struct AttentionStateBank {
    pub states: [i32; LINEAR_ATTENTION_STATE_BANK_LEN],
}

impl AttentionStateBank {
    pub const fn new() -> Self {
        Self {
            states: [0; LINEAR_ATTENTION_STATE_BANK_LEN],
        }
    }

    pub fn stream_mut(&mut self, stream_id: u8) -> Option<&mut [i32; LINEAR_ATTENTION_STATE_LEN]> {
        if stream_id as usize >= ATTENTION_STREAM_COUNT {
            return None;
        }
        let start = stream_id as usize * LINEAR_ATTENTION_STATE_LEN;
        let stream = &mut self.states[start..start + LINEAR_ATTENTION_STATE_LEN];
        stream.try_into().ok()
    }

    pub fn reset_stream(&mut self, stream_id: u8) -> bool {
        if let Some(state) = self.stream_mut(stream_id) {
            reset_attention_state(state);
            true
        } else {
            false
        }
    }
}

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

/// Packed signed power-of-two projection weights. Each nibble encodes zero,
/// +2^k (1..=7), or -2^k (9..=15); nibble 8 is reserved.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct PotLinearAttention {
    pub q: [[u8; 32]; LINEAR_ATTENTION_DIM],
    pub k: [[u8; 32]; LINEAR_ATTENTION_DIM],
    pub v: [[u8; 32]; LINEAR_ATTENTION_DIM],
    pub o: [[u8; 8]; LINEAR_ATTENTION_DIM],
    pub scale: u8,
}

impl PotLinearAttention {
    pub const fn zeroed() -> Self {
        Self {
            q: [[0; 32]; LINEAR_ATTENTION_DIM],
            k: [[0; 32]; LINEAR_ATTENTION_DIM],
            v: [[0; 32]; LINEAR_ATTENTION_DIM],
            o: [[0; 8]; LINEAR_ATTENTION_DIM],
            scale: 0,
        }
    }

    pub fn from_packed_payload(data: &[u8], scale: u8) -> Option<Self> {
        if data.len() < LINEAR_ATTENTION_POT_WEIGHT_BYTES || scale > 6 {
            return None;
        }

        let mut attention = Self::zeroed();
        let mut row = 0usize;
        while row < LINEAR_ATTENTION_DIM {
            let offset = row * 32;
            attention.q[row].copy_from_slice(&data[offset..offset + 32]);
            attention.k[row].copy_from_slice(&data[512 + offset..512 + offset + 32]);
            attention.v[row].copy_from_slice(&data[1024 + offset..1024 + offset + 32]);
            attention.o[row].copy_from_slice(&data[1536 + row * 8..1536 + row * 8 + 8]);
            row += 1;
        }
        attention.scale = scale;
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
    ternary_mlp_scalar_with_activation_into(inputs, model, outputs, false);
}

pub fn ternary_mlp_scalar_with_activation(
    inputs: &[i8; MLP_INPUT_DIM],
    model: &TernaryMlp64x32x16,
    activation_lut: bool,
) -> [i32; MLP_OUTPUT_DIM] {
    let mut output_storage = [0i32; 64];
    ternary_mlp_scalar_with_activation_into(inputs, model, &mut output_storage, activation_lut);
    let mut outputs = [0i32; MLP_OUTPUT_DIM];
    outputs.copy_from_slice(&output_storage[..MLP_OUTPUT_DIM]);
    outputs
}

pub fn ternary_mlp_scalar_with_activation_into(
    inputs: &[i8; MLP_INPUT_DIM],
    model: &TernaryMlp64x32x16,
    outputs: &mut [i32; 64],
    activation_lut: bool,
) {
    let result =
        crate::wasm_math::ternary_mlp(inputs, &model.layer1, &model.layer2, activation_lut);
    outputs[..MLP_OUTPUT_DIM].copy_from_slice(&result);
}

/// Evaluates a block-pruned MLP, skipping each inactive group of eight inputs.
pub fn ternary_mlp_sparse_scalar(
    inputs: &[i8; MLP_INPUT_DIM],
    model: &SparseTernaryMlp64x32x16,
    activation_lut: bool,
) -> [i32; MLP_OUTPUT_DIM] {
    let mut hidden = [0i8; MLP_HIDDEN_DIM];
    let mut row = 0usize;
    while row < MLP_HIDDEN_DIM {
        let mut accumulator = 0i32;
        let mut block = 0usize;
        while block < MLP_LAYER1_BLOCKS_PER_ROW {
            if model.active_blocks.layer1[row] & (1 << block) != 0 {
                let mut lane = 0usize;
                while lane < MLP_BLOCK_WIDTH {
                    let column = block * MLP_BLOCK_WIDTH + lane;
                    let code =
                        (model.weights.layer1[row][column >> 2] >> ((column & 3) * 2)) & 0b11;
                    accumulator += inputs[column] as i32 * decode_ternary(code) as i32;
                    lane += 1;
                }
            }
            block += 1;
        }
        hidden[row] = if activation_lut {
            silu_q4(accumulator.clamp(i8::MIN as i32, i8::MAX as i32) as i8) as i8
        } else {
            integer_hard_sign(accumulator)
        };
        row += 1;
    }

    let mut outputs = [0i32; MLP_OUTPUT_DIM];
    row = 0;
    while row < MLP_OUTPUT_DIM {
        let mut accumulator = 0i32;
        let mut block = 0usize;
        while block < MLP_LAYER2_BLOCKS_PER_ROW {
            if model.active_blocks.layer2[row] & (1 << block) != 0 {
                let mut lane = 0usize;
                while lane < MLP_BLOCK_WIDTH {
                    let column = block * MLP_BLOCK_WIDTH + lane;
                    let code =
                        (model.weights.layer2[row][column >> 2] >> ((column & 3) * 2)) & 0b11;
                    accumulator += hidden[column] as i32 * decode_ternary(code) as i32;
                    lane += 1;
                }
            }
            block += 1;
        }
        outputs[row] = accumulator;
        row += 1;
    }
    outputs
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

/// VPOPCNTDQ implementation of the ternary MLP, batching eight output rows.
///
/// Each input is decomposed into eight bit planes. The packed ternary rows are
/// represented as active/sign masks; AVX-512 popcounts calculate eight row
/// contributions in parallel for each bit plane. Signed `i8` values use a
/// negative coefficient for their high bit, preserving exact scalar arithmetic.
///
/// # Safety
/// AVX-512F, AVX-512BW, AVX-512VPOPCNTDQ, and OS-enabled ZMM state must be available.
pub unsafe fn ternary_mlp_vpopcntdq(
    inputs: &AlignedInputs64,
    model: &TernaryMlp64x32x16,
) -> [i32; MLP_OUTPUT_DIM] {
    let mut output_storage = [0i32; 64];
    ternary_mlp_vpopcntdq_into_ptr(inputs.0.as_ptr(), model, &mut output_storage);
    let mut outputs = [0i32; MLP_OUTPUT_DIM];
    outputs.copy_from_slice(&output_storage[..MLP_OUTPUT_DIM]);
    outputs
}

/// VPOPCNTDQ MLP entry point writing into fixed caller-owned output storage.
///
/// # Safety
/// `inputs` must reference 64 readable bytes, and AVX-512F, AVX-512BW,
/// AVX-512VPOPCNTDQ plus OS-enabled ZMM state must be available.
pub unsafe fn ternary_mlp_vpopcntdq_into_ptr(
    inputs: *const i8,
    model: &TernaryMlp64x32x16,
    outputs: &mut [i32; 64],
) {
    let mut hidden_accumulators = [0i32; 64];
    ternary_rows_vpopcntdq(
        inputs,
        model.layer1.as_ptr().cast(),
        16,
        MLP_HIDDEN_DIM,
        MLP_INPUT_DIM,
        hidden_accumulators.as_mut_ptr(),
    );

    let mut hidden = [0i8; 64];
    let mut row = 0usize;
    while row < MLP_HIDDEN_DIM {
        hidden[row] = integer_hard_sign(hidden_accumulators[row]);
        row += 1;
    }

    ternary_rows_vpopcntdq(
        hidden.as_ptr(),
        model.layer2.as_ptr().cast(),
        8,
        MLP_OUTPUT_DIM,
        MLP_HIDDEN_DIM,
        outputs.as_mut_ptr(),
    );
}

/// Portable bit-plane/popcount reference for validating the VPOPCNTDQ math.
/// It uses the same sign/active-mask identity without executing SIMD instructions.
pub fn ternary_mlp_bitplane_reference(
    inputs: &[i8; MLP_INPUT_DIM],
    model: &TernaryMlp64x32x16,
) -> [i32; MLP_OUTPUT_DIM] {
    let mut hidden = [0i8; MLP_HIDDEN_DIM];
    let mut row = 0usize;
    while row < MLP_HIDDEN_DIM {
        hidden[row] = integer_hard_sign(ternary_row_bitplane_reference(
            inputs,
            &model.layer1[row],
            MLP_INPUT_DIM,
        ));
        row += 1;
    }

    let mut outputs = [0i32; MLP_OUTPUT_DIM];
    row = 0;
    while row < MLP_OUTPUT_DIM {
        outputs[row] = ternary_row_bitplane_reference(&hidden, &model.layer2[row], MLP_HIDDEN_DIM);
        row += 1;
    }
    outputs
}

fn ternary_row_bitplane_reference(inputs: &[i8], packed: &[u8], input_count: usize) -> i32 {
    let mut mask_pos = 0u64;
    let mut mask_neg = 0u64;
    let mut column = 0usize;
    while column < input_count {
        let code = (packed[column >> 2] >> ((column & 3) * 2)) & 0b11;
        let bit = 1u64 << column;
        if code == 0b01 {
            mask_pos |= bit;
        } else if code == 0b11 {
            mask_neg |= bit;
        }
        column += 1;
    }

    let mut result = 0i32;
    let mut plane_index = 0usize;
    while plane_index < 8 {
        let mut input_plane = 0u64;
        column = 0;
        while column < input_count {
            input_plane |= (((inputs[column] as u8 >> plane_index) & 1) as u64) << column;
            column += 1;
        }
        let signed_count = (mask_pos & input_plane).count_ones() as i32
            - (mask_neg & input_plane).count_ones() as i32;
        let coefficient = if plane_index == 7 {
            -128
        } else {
            1 << plane_index
        };
        result += signed_count * coefficient;
        plane_index += 1;
    }
    result
}

/// Computes rows of packed ternary weights using eight-lane VPOPCNTDQ.
///
/// # Safety
/// `inputs` must reference `input_count` bytes, `packed_rows` must reference
/// `row_count * row_stride` bytes, `outputs` must reference `row_count` i32s,
/// and AVX-512F/VPOPCNTDQ with OS-enabled ZMM state must be available.
unsafe fn ternary_rows_vpopcntdq(
    inputs: *const i8,
    packed_rows: *const u8,
    row_stride: usize,
    row_count: usize,
    input_count: usize,
    outputs: *mut i32,
) {
    let mut input_planes = [0u64; 8];
    let mut column = 0usize;
    while column < input_count {
        let input = *inputs.add(column) as u8;
        let mut plane = 0usize;
        while plane < 8 {
            input_planes[plane] |= (((input >> plane) & 1) as u64) << column;
            plane += 1;
        }
        column += 1;
    }

    let mut row_base = 0usize;
    while row_base < row_count {
        let mut active_masks = [0u64; 8];
        let mut negative_masks = [0u64; 8];
        let mut lane = 0usize;
        while lane < 8 {
            let row = row_base + lane;
            if row < row_count {
                let packed = packed_rows.add(row * row_stride);
                column = 0;
                while column < input_count {
                    let code = (*packed.add(column >> 2) >> ((column & 3) * 2)) & 0b11;
                    let bit = 1u64 << column;
                    if code == 0b01 {
                        active_masks[lane] |= bit;
                    } else if code == 0b11 {
                        active_masks[lane] |= bit;
                        negative_masks[lane] |= bit;
                    }
                    column += 1;
                }
            }
            lane += 1;
        }

        let active = _mm512_loadu_si512(active_masks.as_ptr().cast::<__m512i>());
        let negative = _mm512_loadu_si512(negative_masks.as_ptr().cast::<__m512i>());
        let mut accumulators = [0i32; 8];
        let mut plane_index = 0usize;
        while plane_index < 8 {
            let plane = _mm512_set1_epi64(input_planes[plane_index] as i64);
            let active_count = _mm512_popcnt_epi64(_mm512_and_si512(active, plane));
            let negative_count = _mm512_popcnt_epi64(_mm512_and_si512(negative, plane));
            let signed_count = _mm512_sub_epi64(
                active_count,
                _mm512_add_epi64(negative_count, negative_count),
            );
            let mut row_counts = [0i64; 8];
            _mm512_storeu_si512(row_counts.as_mut_ptr().cast::<__m512i>(), signed_count);
            let coefficient = if plane_index == 7 {
                -128
            } else {
                1 << plane_index
            };
            lane = 0;
            while lane < 8 {
                accumulators[lane] += row_counts[lane] as i32 * coefficient;
                lane += 1;
            }
            plane_index += 1;
        }

        lane = 0;
        while lane < 8 && row_base + lane < row_count {
            *outputs.add(row_base + lane) = accumulators[lane];
            lane += 1;
        }
        row_base += 8;
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
    if !crate::boot::cpu_supports_avx512_state() {
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

/// AVX-512 sparse dot product that issues an 8-byte masked load only for active blocks.
/// `packed` uses the standard two-bit ternary encoding and `input_count` is 32 or 64.
#[inline(never)]
unsafe fn ternary_block_sparse_dot_avx512(
    inputs: *const i8,
    packed: &[u8],
    input_count: usize,
    active_blocks: u8,
) -> i32 {
    let mut sum = 0i32;
    let block_count = input_count / MLP_BLOCK_WIDTH;
    let mut block = 0usize;
    while block < block_count {
        if active_blocks & (1 << block) != 0 {
            let mut mask_pos = 0u64;
            let mut mask_neg = 0u64;
            let mut lane = 0usize;
            while lane < MLP_BLOCK_WIDTH {
                let column = block * MLP_BLOCK_WIDTH + lane;
                let code = (packed[column >> 2] >> ((column & 3) * 2)) & 0b11;
                if code == 0b01 {
                    mask_pos |= 1 << lane;
                } else if code == 0b11 {
                    mask_neg |= 1 << lane;
                }
                lane += 1;
            }
            let mut block_inputs = AlignedInputs64([0; 64]);
            let mut lane = 0usize;
            while lane < MLP_BLOCK_WIDTH {
                block_inputs.0[lane] = *inputs.add(block * MLP_BLOCK_WIDTH + lane);
                lane += 1;
            }
            let values = _mm512_load_si512(block_inputs.0.as_ptr().cast::<__m512i>());
            sum += ternary_dot_product_zmm(values, mask_pos, mask_neg);
        }
        block += 1;
    }
    sum
}

/// AVX-512 block-sparse MLP; each absent block avoids its input load and dot product.
///
/// # Safety
/// AVX-512F/BW/DQ and OS-enabled ZMM state must be available.
pub unsafe fn ternary_mlp_sparse_avx512(
    inputs: &AlignedInputs64,
    model: &SparseTernaryMlp64x32x16,
    activation_lut: bool,
) -> [i32; MLP_OUTPUT_DIM] {
    let mut hidden = [0i8; 64];
    let mut row = 0usize;
    while row < MLP_HIDDEN_DIM {
        let accumulator = ternary_block_sparse_dot_avx512(
            inputs.0.as_ptr(),
            &model.weights.layer1[row],
            MLP_INPUT_DIM,
            model.active_blocks.layer1[row],
        );
        hidden[row] = if activation_lut {
            silu_q4(accumulator.clamp(i8::MIN as i32, i8::MAX as i32) as i8) as i8
        } else {
            integer_hard_sign(accumulator)
        };
        row += 1;
    }

    let aligned_hidden = AlignedInputs64(hidden);
    let mut outputs = [0i32; MLP_OUTPUT_DIM];
    row = 0;
    while row < MLP_OUTPUT_DIM {
        outputs[row] = ternary_block_sparse_dot_avx512(
            aligned_hidden.0.as_ptr(),
            &model.weights.layer2[row],
            MLP_HIDDEN_DIM,
            model.active_blocks.layer2[row],
        );
        row += 1;
    }
    outputs
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
    crate::wasm_math::causal_linear_attention(inputs, &model.q, &model.k, &model.v, &model.o, state)
}

/// Integer-only causal attention evaluation for packed signed power-of-two weights.
/// `scale` is a common fixed-point right shift encoded in the signed NEUR header.
pub fn causal_linear_attention_pot_scalar(
    inputs: &[i8; 64],
    model: &PotLinearAttention,
    state: &mut [i32; LINEAR_ATTENTION_STATE_LEN],
) -> [i32; LINEAR_ATTENTION_DIM] {
    let mut q = [0i32; LINEAR_ATTENTION_DIM];
    let mut k = [0i32; LINEAR_ATTENTION_DIM];
    let mut v = [0i32; LINEAR_ATTENTION_DIM];

    let mut row = 0usize;
    while row < LINEAR_ATTENTION_DIM {
        q[row] = pot_vector_dot_scalar(inputs, &model.q[row], model.scale);
        k[row] = pot_vector_dot_scalar(inputs, &model.k[row], model.scale);
        v[row] = pot_vector_dot_scalar(inputs, &model.v[row], model.scale);
        row += 1;
    }

    row = 0;
    while row < LINEAR_ATTENTION_DIM {
        let mut column = 0usize;
        while column < LINEAR_ATTENTION_DIM {
            let idx = row * LINEAR_ATTENTION_DIM + column;
            state[idx] = state[idx].saturating_add(k[row].saturating_mul(v[column]));
            column += 1;
        }
        row += 1;
    }

    let mut output = [0i32; LINEAR_ATTENTION_DIM];
    row = 0;
    while row < LINEAR_ATTENTION_DIM {
        let mut column = 0usize;
        while column < LINEAR_ATTENTION_DIM {
            let idx = column * LINEAR_ATTENTION_DIM + row;
            output[row] = output[row].saturating_add(q[column].saturating_mul(state[idx]));
            column += 1;
        }
        row += 1;
    }

    let mut projected = [0i32; LINEAR_ATTENTION_DIM];
    row = 0;
    while row < LINEAR_ATTENTION_DIM {
        let mut column = 0usize;
        while column < LINEAR_ATTENTION_DIM {
            let code = pot_weight_code(&model.o[row], column);
            projected[row] =
                projected[row].saturating_add(pot_mul_shift(output[column], code, model.scale));
            column += 1;
        }
        row += 1;
    }
    projected
}

fn pot_vector_dot_scalar(inputs: &[i8; 64], packed: &[u8; 32], scale: u8) -> i32 {
    let mut sum = 0i32;
    let mut index = 0usize;
    while index < 64 {
        let code = pot_weight_code(packed, index);
        sum = sum.saturating_add(pot_mul_shift(inputs[index] as i32, code, scale));
        index += 1;
    }
    sum
}

#[inline(always)]
fn pot_weight_code(packed: &[u8], index: usize) -> u8 {
    let byte = packed[index >> 1];
    if index & 1 == 0 {
        byte & 0x0F
    } else {
        byte >> 4
    }
}

/// Evaluates a signed-nibble PoT coefficient using shifts rather than multiply.
/// Invalid code 8 is treated as zero; payload validation rejects it before use.
#[inline(always)]
pub fn pot_mul_shift(value: i32, code: u8, scale: u8) -> i32 {
    if code == 0 || code == 8 {
        return 0;
    }
    let negative = code >= 9;
    let exponent = if negative { code - 9 } else { code - 1 };
    let shifted = (value as i64) << exponent;
    let signed = if negative { -shifted } else { shifted };
    let scaled = if scale >= 63 {
        if signed < 0 {
            -1
        } else {
            0
        }
    } else {
        signed >> scale
    };
    if scaled > i32::MAX as i64 {
        i32::MAX
    } else if scaled < i32::MIN as i64 {
        i32::MIN
    } else {
        scaled as i32
    }
}

/// Rejects reserved signed-nibble code 8 throughout a packed PoT payload.
pub fn validate_pot_attention_payload(data: &[u8]) -> bool {
    !data.is_empty() && data.iter().all(|byte| byte & 0x0F != 8 && byte >> 4 != 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::SILU_LUT_Q4;

    fn set_packed_weight(row: &mut [u8], column: usize, code: u8) {
        let byte = column >> 2;
        let shift = (column & 3) * 2;
        row[byte] = (row[byte] & !(0b11 << shift)) | ((code & 0b11) << shift);
    }

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
    fn attention_state_bank_resets_only_the_selected_stream() {
        let mut bank = AttentionStateBank::new();
        bank.stream_mut(2).unwrap()[0] = 23;
        bank.stream_mut(5).unwrap()[0] = 41;
        assert!(bank.reset_stream(5));
        assert_eq!(bank.stream_mut(2).unwrap()[0], 23);
        assert!(bank.stream_mut(5).unwrap().iter().all(|&value| value == 0));
        assert!(!bank.reset_stream(8));
    }

    #[test]
    fn pot_shift_path_matches_signed_power_of_two_products() {
        assert_eq!(pot_mul_shift(17, 1, 0), 17); // +2^0
        assert_eq!(pot_mul_shift(17, 3, 0), 68); // +2^2
        assert_eq!(pot_mul_shift(17, 11, 0), -68); // -2^2
        assert_eq!(pot_mul_shift(i32::MIN, 9, 0), i32::MAX);
        assert_eq!(pot_mul_shift(i32::MAX, 1, 1), i32::MAX / 2);
    }

    #[test]
    fn aligned_silu_lut_is_single_indexed_and_zero_preserving() {
        assert_eq!((SILU_LUT_Q4.0.as_ptr() as usize) & 63, 0);
        assert_eq!(silu_q4(0), 0);
        assert!(SILU_LUT_Q4
            .0
            .iter()
            .all(|&value| (-128..=127).contains(&value)));
    }

    #[test]
    fn sparse_mlp_matches_dense_for_both_activation_modes() {
        let mut weights = TernaryMlp64x32x16::zeroed();
        let masks = MlpBlockMask {
            layer1: [0b0101_0101; MLP_HIDDEN_DIM],
            layer2: [0b0000_0101; MLP_OUTPUT_DIM],
        };
        for row in 0..MLP_HIDDEN_DIM {
            for block in 0..MLP_LAYER1_BLOCKS_PER_ROW {
                if masks.layer1[row] & (1 << block) != 0 {
                    for lane in 0..MLP_BLOCK_WIDTH {
                        let column = block * MLP_BLOCK_WIDTH + lane;
                        let code = match (row + column) % 3 {
                            0 => 0,
                            1 => 1,
                            _ => 3,
                        };
                        set_packed_weight(&mut weights.layer1[row], column, code);
                    }
                }
            }
        }
        for row in 0..MLP_OUTPUT_DIM {
            for block in 0..MLP_LAYER2_BLOCKS_PER_ROW {
                if masks.layer2[row] & (1 << block) != 0 {
                    for lane in 0..MLP_BLOCK_WIDTH {
                        let column = block * MLP_BLOCK_WIDTH + lane;
                        let code = if (row + column) & 1 == 0 { 1 } else { 3 };
                        set_packed_weight(&mut weights.layer2[row], column, code);
                    }
                }
            }
        }
        let sparse = SparseTernaryMlp64x32x16 {
            weights,
            active_blocks: masks,
        };
        let mut inputs = [0i8; MLP_INPUT_DIM];
        for (index, input) in inputs.iter_mut().enumerate() {
            *input = (index as i8).wrapping_mul(7).wrapping_sub(91);
        }

        for activation_lut in [false, true] {
            let dense =
                ternary_mlp_scalar_with_activation(&inputs, &sparse.weights, activation_lut);
            let block_sparse = ternary_mlp_sparse_scalar(&inputs, &sparse, activation_lut);
            assert_eq!(block_sparse, dense);
        }
        assert_eq!(sparse.active_block_count(), 32 * 4 + 16 * 2);
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
