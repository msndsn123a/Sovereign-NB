//! 2-bit ternary weight packing and 64-bit mask extraction utilities.
//!
//! Encoding specification:
//! - `00b`: Zero / Inactive (`0`)
//! - `01b`: Positive (`+1`)
//! - `11b`: Negative (`-1`)
//!
//! Each byte (`u8`) packs 4 ternary weights in little-endian bit order
//! (weight 0 in bits `1..=0`, weight 1 in bits `3..=2`, weight 2 in bits `5..=4`,
//! weight 3 in bits `7..=6`).

/// 2-bit code for zero / inactive weight (`0`).
pub const WEIGHT_ZERO: u8 = 0b00;
/// 2-bit code for positive weight (`+1`).
pub const WEIGHT_POS: u8 = 0b01;
/// 2-bit code for negative weight (`-1`).
pub const WEIGHT_NEG: u8 = 0b11;

pub const NEUR_MAGIC: u32 = 0x4E45_5552;
pub const NEUR_BASE_HEADER_SIZE: usize = 16;
pub const NEUR_SIGNATURE_SIZE: usize = 64;
pub const NEUR_HEADER_SIZE: usize = NEUR_BASE_HEADER_SIZE + NEUR_SIGNATURE_SIZE;
pub const NEUR_INPUT_DIM: u32 = 64;
pub const NEUR_HIDDEN_DIM: u8 = 32;
pub const NEUR_OUTPUT_DIM: u16 = 16;
pub const NEUR_ATTN_DIM: u8 = 16;
pub const NEUR_MODEL_MLP: u8 = 0;
pub const NEUR_MODEL_CAUSAL_LINEAR_ATTENTION: u8 = 1;
pub const NEUR_QUANT_POT: u8 = 1;
pub const NEUR_FLAG_POT: u8 = 0x40;
pub const NEUR_FLAG_BLOCK_SPARSE: u8 = 0x40;
pub const NEUR_FLAG_MULTI_STREAM: u8 = 0x80;
pub const NEUR_FLAG_ACTIVATION_LUT: u16 = 0x8000;
pub const NEUR_ATTENTION_POT_PAYLOAD_BYTES: usize = 1664;
pub const NEUR_LAYER1_BYTES: usize = NEUR_HIDDEN_DIM as usize * 16;
pub const NEUR_LAYER2_BYTES: usize = NEUR_OUTPUT_DIM as usize * 8;
pub const NEUR_ATTENTION_QK_BYTES: usize = NEUR_INPUT_DIM as usize * NEUR_ATTN_DIM as usize / 4;
pub const NEUR_ATTENTION_V_BYTES: usize = NEUR_INPUT_DIM as usize * NEUR_ATTN_DIM as usize / 4;
pub const NEUR_ATTENTION_O_BYTES: usize = NEUR_OUTPUT_DIM as usize * NEUR_ATTN_DIM as usize / 4;
pub const NEUR_ATTENTION_PAYLOAD_BYTES: usize =
    NEUR_ATTENTION_QK_BYTES * 3 + NEUR_ATTENTION_O_BYTES;
pub const NEUR_MLP_PAYLOAD_BYTES: usize = NEUR_LAYER1_BYTES + NEUR_LAYER2_BYTES;
pub const NEUR_MLP_BLOCK_MASK_BYTES: usize = NEUR_HIDDEN_DIM as usize + NEUR_OUTPUT_DIM as usize;
pub const NEUR_MLP_SPARSE_PAYLOAD_BYTES: usize = NEUR_MLP_PAYLOAD_BYTES + NEUR_MLP_BLOCK_MASK_BYTES;
pub const NEUR_MLP_SHARD_SIZE: usize = NEUR_HEADER_SIZE + NEUR_MLP_PAYLOAD_BYTES;
pub const NEUR_ATTENTION_SHARD_SIZE: usize = NEUR_HEADER_SIZE + NEUR_ATTENTION_PAYLOAD_BYTES;
pub const NEUR_QUANT_TERNARY: u8 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeurHeader {
    pub version: u32,
    pub input_dim: u32,
    pub hidden_dim: u8,
    pub output_dim: u16,
    pub quant_type: u8,
    pub model_type: u8,
    pub attn_dim: u16,
    pub multi_stream: bool,
    pub pot_scale: u8,
    pub block_sparse: bool,
    pub activation_lut: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeurHeaderError {
    Truncated,
    BadMagic,
    UnsupportedInputDimension,
    InvalidOutputDimension,
    InvalidHiddenDimension,
    UnsupportedQuantization,
    NonzeroReservedByte,
}

/// Parses and validates the fixed 16-byte NEUR metadata prefix.
///
/// Layout: magic[0..4], version[4..8], input_dim[8..12], model_type[12],
/// output_dim[13..15] little-endian (upper bits carry PoT scale and LUT flag),
/// and dimension/flags in byte 15. Feature flags are authenticated as part of
/// the existing signed 16-byte prefix.
pub fn parse_neur_header(data: &[u8]) -> Result<NeurHeader, NeurHeaderError> {
    if data.len() < NEUR_BASE_HEADER_SIZE {
        return Err(NeurHeaderError::Truncated);
    }

    let magic = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
    if magic != NEUR_MAGIC {
        return Err(NeurHeaderError::BadMagic);
    }

    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let input_dim = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let model_type = data[12];
    let output_meta = u16::from_le_bytes([data[13], data[14]]);
    let output_dim = output_meta & 0x0FFF;
    let pot_scale = ((output_meta >> 12) & 0x07) as u8;
    let activation_lut = output_meta & NEUR_FLAG_ACTIVATION_LUT != 0;
    let header_flags = data[15];
    let hidden_or_attn_dim = header_flags & 0x3F;
    let multi_stream = header_flags & NEUR_FLAG_MULTI_STREAM != 0;
    let is_attention = model_type == NEUR_MODEL_CAUSAL_LINEAR_ATTENTION;
    let is_mlp = model_type == NEUR_MODEL_MLP;
    let feature_flag = header_flags & NEUR_FLAG_POT != 0;
    let quant_type = u8::from(is_attention && feature_flag);
    let block_sparse = is_mlp && feature_flag;

    if input_dim != NEUR_INPUT_DIM {
        return Err(NeurHeaderError::UnsupportedInputDimension);
    }
    if output_dim != NEUR_OUTPUT_DIM {
        return Err(NeurHeaderError::InvalidOutputDimension);
    }
    if pot_scale > 6 || (quant_type == NEUR_QUANT_TERNARY && pot_scale != 0) {
        return Err(NeurHeaderError::UnsupportedQuantization);
    }

    if !(is_attention || is_mlp) {
        return Err(NeurHeaderError::UnsupportedQuantization);
    }
    if quant_type == NEUR_QUANT_POT && !is_attention {
        return Err(NeurHeaderError::UnsupportedQuantization);
    }
    if multi_stream && !is_attention {
        return Err(NeurHeaderError::NonzeroReservedByte);
    }
    if activation_lut && !is_mlp {
        return Err(NeurHeaderError::NonzeroReservedByte);
    }

    let hidden_dim = if is_attention {
        if hidden_or_attn_dim != NEUR_ATTN_DIM {
            return Err(NeurHeaderError::InvalidHiddenDimension);
        }
        hidden_or_attn_dim
    } else {
        if hidden_or_attn_dim != NEUR_HIDDEN_DIM {
            return Err(NeurHeaderError::InvalidHiddenDimension);
        }
        hidden_or_attn_dim
    };

    Ok(NeurHeader {
        version,
        input_dim,
        hidden_dim,
        output_dim,
        quant_type,
        model_type,
        attn_dim: if is_attention {
            hidden_or_attn_dim as u16
        } else {
            NEUR_ATTN_DIM as u16
        },
        multi_stream,
        pot_scale,
        block_sparse,
        activation_lut,
    })
}

/// Checks a packed ternary array and rejects the unused `10b` code.
pub fn validate_ternary_payload(data: &[u8]) -> bool {
    if data.is_empty() {
        return false;
    }

    let mut index = 0usize;
    while index < data.len() {
        let byte = data[index];
        let mut lane = 0usize;
        while lane < 4 {
            if ((byte >> (lane * 2)) & 0b11) == 0b10 {
                return false;
            }
            lane += 1;
        }
        index += 1;
    }
    true
}

/// Validates packed signed-nibble PoT attention weights; nibble 8 is reserved.
pub fn validate_pot_payload(data: &[u8]) -> bool {
    !data.is_empty() && data.iter().all(|byte| byte & 0x0F != 8 && byte >> 4 != 8)
}

/// Encodes a single ternary weight (`-1`, `0`, `+1`) into its 2-bit representation.
#[inline(always)]
pub const fn encode_ternary(weight: i8) -> u8 {
    match weight {
        1 => WEIGHT_POS,
        -1 => WEIGHT_NEG,
        _ => WEIGHT_ZERO,
    }
}

/// Decodes a 2-bit code (`00b`, `01b`, `11b`) back into a signed ternary weight (`i8`).
#[inline(always)]
pub const fn decode_ternary(code: u8) -> i8 {
    match code & 0b11 {
        WEIGHT_POS => 1,
        WEIGHT_NEG => -1,
        _ => 0,
    }
}

/// Packs 64 ternary weights (`{-1, 0, 1}`) into 16 bytes (4 weights per byte).
#[inline]
pub const fn pack_weights_64(weights: &[i8; 64]) -> [u8; 16] {
    let mut packed = [0u8; 16];
    let mut byte_idx = 0usize;
    while byte_idx < 16 {
        let base = byte_idx * 4;
        let b0 = encode_ternary(weights[base]);
        let b1 = encode_ternary(weights[base + 1]) << 2;
        let b2 = encode_ternary(weights[base + 2]) << 4;
        let b3 = encode_ternary(weights[base + 3]) << 6;
        packed[byte_idx] = b0 | b1 | b2 | b3;
        byte_idx += 1;
    }
    packed
}

/// Unpacks a 16-byte packed weight array (64 ternary weights) into two 64-bit masks:
/// - `mask_pos`: bit `i` is `1` iff weight `i` is `+1` (`01b`)
/// - `mask_neg`: bit `i` is `1` iff weight `i` is `-1` (`11b`)
#[inline]
pub const fn unpack_masks_64(packed: &[u8; 16]) -> (u64, u64) {
    let mut mask_pos: u64 = 0;
    let mut mask_neg: u64 = 0;

    let mut byte_idx = 0usize;
    while byte_idx < 16 {
        let byte = packed[byte_idx];
        let mut sub = 0usize;
        while sub < 4 {
            let code = (byte >> (sub * 2)) & 0b11;
            let bit_index = byte_idx * 4 + sub;
            let bit = 1u64 << bit_index;

            // Pure bitwise classification:
            // 01b -> positive (+1)
            // 11b -> negative (-1)
            if code == WEIGHT_POS {
                mask_pos |= bit;
            } else if code == WEIGHT_NEG {
                mask_neg |= bit;
            }
            sub += 1;
        }
        byte_idx += 1;
    }

    (mask_pos, mask_neg)
}

/// Extracts `(mask_pos, mask_neg)` from a packed weight byte slice (must have at least 16 bytes).
#[inline]
pub fn extract_masks(packed: &[u8]) -> (u64, u64) {
    assert!(packed.len() >= 16);
    let mut block = [0u8; 16];
    let mut i = 0usize;
    while i < 16 {
        block[i] = packed[i];
        i += 1;
    }
    unpack_masks_64(&block)
}
