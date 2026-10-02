//! Offline Raw Shard Builder Tool for Neural-Box Core.
//!
//! Encodes and Ed25519-signs ternary model weights into NEUR v2 shards.

use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;
use ed25519_dalek::{Signer, SigningKey};

pub const MAGIC_NEUR: [u8; 4] = *b"NEUR"; // 0x4E455552
pub const SECTOR_SIZE: usize = 512;
const ATTENTION_WEIGHT_BYTES: usize = 832;
const ATTENTION_POT_WEIGHT_BYTES: usize = 1664;
const MLP_WEIGHT_BYTES: usize = 640;
const MLP_BLOCK_MASK_BYTES: usize = 48;
const BASE_HEADER_SIZE: usize = 16;
const SIGNATURE_SIZE: usize = 64;
const SIGNED_HEADER_SIZE: usize = BASE_HEADER_SIZE + SIGNATURE_SIZE;
const FLAG_POT: u8 = 0x40;
const FLAG_BLOCK_SPARSE: u8 = 0x40;
const FLAG_MULTI_STREAM: u8 = 0x80;
const FLAG_ACTIVATION_LUT: u16 = 0x8000;
const TEST_SIGNING_SEED: [u8; 32] = [
    0x9D, 0x61, 0xB1, 0x9D, 0xEF, 0xFD, 0x5A, 0x60, 0xBA, 0x84, 0x4A, 0xF4, 0x92, 0xEC, 0x2C,
    0xC4, 0x44, 0x49, 0xC5, 0x69, 0x7B, 0x32, 0x69, 0x19, 0x70, 0x3B, 0xAC, 0x03, 0x1C, 0xAE,
    0x7F, 0x60,
];

fn sign_payload(sector: &mut [u8], payload_size: usize, signing_key: &SigningKey) {
    let payload_end = SIGNED_HEADER_SIZE + payload_size;
    let mut message = Vec::with_capacity(BASE_HEADER_SIZE + payload_size);
    message.extend_from_slice(&sector[..BASE_HEADER_SIZE]);
    message.extend_from_slice(&sector[SIGNED_HEADER_SIZE..payload_end]);
    let signature = signing_key.sign(&message).to_bytes();
    sector[BASE_HEADER_SIZE..SIGNED_HEADER_SIZE].copy_from_slice(&signature);
}

fn sign_unsigned_shard(
    input: &[u8],
    block_size: usize,
    signing_key: &SigningKey,
) -> Result<(Vec<u8>, &'static str), String> {
    if !(512..=4096).contains(&block_size) || !block_size.is_power_of_two() {
        return Err("block size must be a power of two from 512 through 4096".to_string());
    }
    if input.len() < BASE_HEADER_SIZE || input[..4] != MAGIC_NEUR {
        return Err("unsigned input must have a 16-byte NEUR metadata prefix".to_string());
    }
    let model_type = input[12];
    let has_feature_flag = input[15] & FLAG_POT != 0;
    let (payload_size, model_name) = match model_type {
        0 if has_feature_flag => (MLP_WEIGHT_BYTES + MLP_BLOCK_MASK_BYTES, "mlp"),
        0 => (MLP_WEIGHT_BYTES, "mlp"),
        1 if has_feature_flag => (ATTENTION_POT_WEIGHT_BYTES, "attention"),
        1 => (ATTENTION_WEIGHT_BYTES, "attention"),
        _ => return Err(format!("unsupported unsigned NEUR model_type={model_type}")),
    };
    if input.len() < BASE_HEADER_SIZE + payload_size {
        return Err("unsigned NEUR input is truncated".to_string());
    }
    let signed_size = SIGNED_HEADER_SIZE + payload_size;
    let padded_size = signed_size.div_ceil(block_size) * block_size;
    let mut shard = vec![0u8; padded_size];
    shard[..BASE_HEADER_SIZE].copy_from_slice(&input[..BASE_HEADER_SIZE]);
    shard[4..8].copy_from_slice(&2u32.to_le_bytes());
    shard[SIGNED_HEADER_SIZE..signed_size]
        .copy_from_slice(&input[BASE_HEADER_SIZE..BASE_HEADER_SIZE + payload_size]);
    sign_payload(&mut shard, payload_size, signing_key);
    Ok((shard, model_name))
}

/// Encodes a single ternary weight into 2 bits:
/// - 00b: 0 (Zero)
/// - 01b: +1 (Positive)
/// - 11b: -1 (Negative)
pub fn encode_ternary(w: i8) -> u8 {
    match w {
        1 => 0b01,
        -1 => 0b11,
        _ => 0b00,
    }
}

/// Packs 64 ternary weights into 16 bytes (4 weights per byte).
pub fn pack_weights_64(weights: &[i8; 64]) -> [u8; 16] {
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

/// Builds and signs a 64 -> 32 -> 16 ternary MLP shard.
pub fn build_shard_sector(
    version: u32,
    block_size: usize,
    layer1_weights: &[i8; 64],
    signing_key: &SigningKey,
    prune_blocks: bool,
    activation_lut: bool,
) -> Vec<u8> {
    assert!((512..=4096).contains(&block_size) && block_size.is_power_of_two());
    const INPUT_DIM: u32 = 64;
    const HIDDEN_DIM: u8 = 32;
    const OUTPUT_DIM: u16 = 16;
    const LAYER1_BYTES: usize = HIDDEN_DIM as usize * 16;
    const LAYER2_BYTES: usize = OUTPUT_DIM as usize * 8;
    let base_model_payload_size = LAYER1_BYTES + LAYER2_BYTES;
    let model_payload_size = base_model_payload_size
        + if prune_blocks { MLP_BLOCK_MASK_BYTES } else { 0 };
    let payload_size = SIGNED_HEADER_SIZE + model_payload_size;
    let padded_size = payload_size.div_ceil(block_size) * block_size;
    let mut sector = vec![0u8; padded_size];

    // 1. Magic bytes: "NEUR" (0x4E455552)
    sector[0..4].copy_from_slice(&MAGIC_NEUR);

    // 2. Model Version (u32 little-endian)
    sector[4..8].copy_from_slice(&version.to_le_bytes());

    // 3. Input Dimension (u32 little-endian, e.g. 64)
    sector[8..12].copy_from_slice(&INPUT_DIM.to_le_bytes());

    // 4. Weight Quantization Type (u8, 0 = Ternary)
    sector[12] = 0; // Ternary {-1, 0, +1}

    // Bytes 13..15 store output_dim (u16 LE) then hidden_dim (u8).
    let output_meta = OUTPUT_DIM | if activation_lut { FLAG_ACTIVATION_LUT } else { 0 };
    sector[13..15].copy_from_slice(&output_meta.to_le_bytes());
    sector[15] = HIDDEN_DIM | if prune_blocks { FLAG_BLOCK_SPARSE } else { 0 };

    // Layer 1: 32 rows of 64 ternary weights.
    let packed_layer1 = pack_weights_64(layer1_weights);
    let mut row = 0usize;
    while row < HIDDEN_DIM as usize {
        let start = SIGNED_HEADER_SIZE + row * packed_layer1.len();
        sector[start..start + packed_layer1.len()].copy_from_slice(&packed_layer1);
        row += 1;
    }

    // Layer 2: 16 rows of 32 positive ternary weights.
    let layer2_start = SIGNED_HEADER_SIZE + LAYER1_BYTES;
    let packed_positive = [0x55u8; 8];
    row = 0;
    while row < OUTPUT_DIM as usize {
        let start = layer2_start + row * packed_positive.len();
        sector[start..start + packed_positive.len()].copy_from_slice(&packed_positive);
        row += 1;
    }

    if prune_blocks {
        let mask_start = SIGNED_HEADER_SIZE + base_model_payload_size;
        let layer1_masks = 0b0101_0101u8;
        let layer2_masks = 0b0000_0101u8;
        for row in 0..HIDDEN_DIM as usize {
            sector[mask_start + row] = layer1_masks;
            for block in 0..8 {
                if layer1_masks & (1 << block) == 0 {
                    let start = SIGNED_HEADER_SIZE + row * 16 + block * 2;
                    sector[start..start + 2].fill(0);
                }
            }
        }
        for row in 0..OUTPUT_DIM as usize {
            sector[mask_start + HIDDEN_DIM as usize + row] = layer2_masks;
            for block in 0..4 {
                if layer2_masks & (1 << block) == 0 {
                    let start = SIGNED_HEADER_SIZE + LAYER1_BYTES + row * 8 + block * 2;
                    sector[start..start + 2].fill(0);
                }
            }
        }
    }

    sign_payload(&mut sector, model_payload_size, signing_key);
    // The remainder of the final LBA block is zero-padded.
    sector
}

/// Builds and signs a causal linear-attention test shard with all-positive Q/K/V
/// weights and an identity O projection.
pub fn build_attention_shard_sector(
    version: u32,
    block_size: usize,
    signing_key: &SigningKey,
    multi_stream: bool,
) -> Vec<u8> {
    assert!((512..=4096).contains(&block_size) && block_size.is_power_of_two());
    let payload_size = SIGNED_HEADER_SIZE + ATTENTION_WEIGHT_BYTES;
    let padded_size = payload_size.div_ceil(block_size) * block_size;
    let mut sector = vec![0u8; padded_size];
    sector[0..4].copy_from_slice(&MAGIC_NEUR);
    sector[4..8].copy_from_slice(&version.to_le_bytes());
    sector[8..12].copy_from_slice(&64u32.to_le_bytes());
    sector[12] = 1; // Causal linear attention model type.
    sector[13..15].copy_from_slice(&16u16.to_le_bytes());
    sector[15] = 16 | if multi_stream { FLAG_MULTI_STREAM } else { 0 };

    // 00 01 01 01... repeated: every projection weight is +1.
    sector[SIGNED_HEADER_SIZE..SIGNED_HEADER_SIZE + ATTENTION_WEIGHT_BYTES].fill(0x55);
    let output_start = SIGNED_HEADER_SIZE + (3 * 256);
    sector[output_start..output_start + 64].fill(0);
    for row in 0..16 {
        let row_start = output_start + row * 4;
        sector[row_start + (row >> 2)] |= 0x01 << ((row & 3) * 2);
    }
    sign_payload(&mut sector, ATTENTION_WEIGHT_BYTES, signing_key);
    sector
}

/// Builds a signed PoT attention model using 4-bit signed-shift coefficients.
pub fn build_pot_attention_shard_sector(
    version: u32,
    block_size: usize,
    signing_key: &SigningKey,
    multi_stream: bool,
    pot_scale: u8,
) -> Vec<u8> {
    assert!((512..=4096).contains(&block_size) && block_size.is_power_of_two());
    assert!(pot_scale <= 6);
    let payload_size = SIGNED_HEADER_SIZE + ATTENTION_POT_WEIGHT_BYTES;
    let padded_size = payload_size.div_ceil(block_size) * block_size;
    let mut sector = vec![0u8; padded_size];
    sector[0..4].copy_from_slice(&MAGIC_NEUR);
    sector[4..8].copy_from_slice(&version.to_le_bytes());
    sector[8..12].copy_from_slice(&64u32.to_le_bytes());
    sector[12] = 1;
    let output_meta = 16u16 | ((pot_scale as u16) << 12);
    sector[13..15].copy_from_slice(&output_meta.to_le_bytes());
    sector[15] = 16 | FLAG_POT | if multi_stream { FLAG_MULTI_STREAM } else { 0 };

    let payload = &mut sector[SIGNED_HEADER_SIZE..payload_size];
    // PoT nibble 1 represents +2^0. Q/K/V are all positive; O is identity.
    payload[..ATTENTION_POT_WEIGHT_BYTES].fill(0x11);
    let output_start = 3 * 512;
    payload[output_start..].fill(0);
    for row in 0..16 {
        let nibble_index = row * 16 + row;
        let byte_index = output_start + nibble_index / 2;
        if nibble_index & 1 == 0 {
            payload[byte_index] |= 0x01;
        } else {
            payload[byte_index] |= 0x10;
        }
    }
    sign_payload(&mut sector, ATTENTION_POT_WEIGHT_BYTES, signing_key);
    sector
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out_path = "dist/production_shard.bin".to_string();
    let mut block_size = SECTOR_SIZE;
    let mut model_type = "mlp".to_string();
    let mut variant = "pattern".to_string();
    let mut quantization = "ternary".to_string();
    let mut multi_stream = false;
    let mut pot_scale = 0u8;
    let mut prune_blocks = false;
    let mut activation_lut = false;
    let mut key_file = None;
    let mut unsigned_input = None;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" || args[i] == "-o" {
            if i + 1 < args.len() {
                out_path = args[i + 1].clone();
                i += 1;
            }
        } else if args[i] == "--block-size" {
            if i + 1 < args.len() {
                block_size = args[i + 1].parse()?;
                i += 1;
            }
        } else if args[i] == "--model" {
            if i + 1 < args.len() {
                model_type = args[i + 1].clone();
                i += 1;
            }
        } else if args[i] == "--variant" {
            if i + 1 < args.len() {
                variant = args[i + 1].clone();
                i += 1;
            }
        } else if args[i] == "--quant" {
            if i + 1 < args.len() {
                quantization = args[i + 1].clone();
                i += 1;
            }
        } else if args[i] == "--pot-scale" {
            if i + 1 < args.len() {
                pot_scale = args[i + 1].parse()?;
                i += 1;
            }
        } else if args[i] == "--multi-stream" {
            multi_stream = true;
        } else if args[i] == "--prune-blocks" {
            prune_blocks = true;
        } else if args[i] == "--activation-lut" {
            activation_lut = true;
        } else if args[i] == "--key-file" {
            if i + 1 < args.len() {
                key_file = Some(args[i + 1].clone());
                i += 1;
            }
        } else if args[i] == "--unsigned-input" {
            if i + 1 < args.len() {
                unsigned_input = Some(args[i + 1].clone());
                i += 1;
            }
        }
        i += 1;
    }

    println!("[PAYLOAD_BUILDER]: Initializing raw model shard packaging...");

    // Generate standard deterministic reference ternary weights matching appliance verification
    let mut weights = [0i8; 64];
    let weight_pattern: [i8; 4] = [1, -1, 0, 1];
    let mut w_idx = 0usize;
    while w_idx < 64 {
        weights[w_idx] = weight_pattern[(w_idx + (w_idx >> 2)) & 3];
        w_idx += 1;
    }
    if variant == "zero" {
        weights.fill(0);
    } else if variant != "pattern" {
        return Err("--variant must be pattern or zero".into());
    }

    const VERSION: u32 = 2;
    const INPUT_DIM: u32 = 64;

    let seed = if let Some(path) = key_file.as_ref() {
        let key_bytes = std::fs::read(path)?;
        key_bytes
            .try_into()
            .map_err(|_| "Ed25519 signing seed file must contain exactly 32 bytes")?
    } else {
        println!("  - Signing Key:        deterministic test key (development only)");
        TEST_SIGNING_SEED
    };
    let signing_key = SigningKey::from_bytes(&seed);

    let (sector, model_type) = if let Some(input_path) = unsigned_input {
        let input = std::fs::read(input_path)?;
        let (signed, detected_model) = sign_unsigned_shard(&input, block_size, &signing_key)?;
        (signed, detected_model.to_string())
    } else {
        let sector = match model_type.as_str() {
            "mlp" if variant != "pattern" && variant != "zero" => {
                return Err("--variant must be pattern or zero".into())
            }
            "mlp" if quantization != "ternary" || multi_stream || pot_scale != 0 => {
                return Err("multi-stream and PoT options are currently attention-only".into())
            }
            "mlp" => build_shard_sector(
                VERSION,
                block_size,
                &weights,
                &signing_key,
                prune_blocks,
                activation_lut,
            ),
            "attention" if prune_blocks || activation_lut => {
                return Err("block pruning and activation LUT options are currently MLP-only".into())
            }
            "attention" if variant != "pattern" => {
                return Err("--variant zero is currently MLP-only".into())
            }
            "attention" if quantization == "ternary" => {
                if pot_scale != 0 {
                    return Err("--pot-scale requires --quant pot".into());
                }
                build_attention_shard_sector(VERSION, block_size, &signing_key, multi_stream)
            }
            "attention" if quantization == "pot" => build_pot_attention_shard_sector(
                VERSION,
                block_size,
                &signing_key,
                multi_stream,
                pot_scale,
            ),
            "attention" => return Err("--quant must be ternary or pot".into()),
            _ => {
                return Err(format!("unsupported model type: {model_type} (use mlp or attention)").into())
            }
        };
        (sector, model_type)
    };

    // Ensure output directory exists
    if let Some(parent) = Path::new(&out_path).parent() {
        create_dir_all(parent)?;
    }

    let mut file = File::create(&out_path)?;
    file.write_all(&sector)?;

    println!("[PAYLOAD_BUILDER]: Packaging Complete.");
    println!("  - Target File:        {}", out_path);
    println!("  - Magic Header:       0x4E455552 (\"NEUR\")");
    println!("  - Model Version:      {}", VERSION);
    println!("  - Header Size:        {} bytes (Ed25519 signature included)", SIGNED_HEADER_SIZE);
    println!("  - Public Key:         {:02x?}", signing_key.verifying_key().to_bytes());
    println!("  - Input Dimension:    {} elements", INPUT_DIM);
    println!("  - Model Type:         {}", model_type);
    println!("  - Test Variant:       {}", variant);
    println!("  - Multi-stream:       {}", multi_stream);
    println!("  - Quantization:       {}", quantization);
    println!("  - Block pruning:      {}", prune_blocks);
    println!("  - Activation LUT:    {}", activation_lut);
    println!(
        "  - Hidden/Attention:   {} elements",
        if model_type == "attention" { 16 } else { 32 }
    );
    println!("  - Output Dimension:   16 elements");
    println!("  - LBA Block Size:     {} bytes", block_size);
    println!(
        "  - PoT Scale:          {} (signed right-shift metadata)",
        pot_scale
    );
    println!(
        "  - Payload Size:       {} bytes (LBA-block padded)",
        sector.len()
    );
    if model_type == "mlp" {
        println!("  - Layer 1 Size:       512 bytes");
        println!("  - Layer 2 Size:       128 bytes");
        if prune_blocks {
            println!("  - Active Masks:       {} bytes", MLP_BLOCK_MASK_BYTES);
        }
    } else {
        let weight_bytes = if quantization == "pot" {
            ATTENTION_POT_WEIGHT_BYTES
        } else {
            ATTENTION_WEIGHT_BYTES
        };
        println!("  - Attention Size:     {} bytes", weight_bytes);
    }

    Ok(())
}
