//! Offline Raw Shard Builder Tool for Neural-Box Core.
//!
//! Encodes ternary model weights {-1, 0, 1} into hardware-ingestible LBA blocks.

use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;

pub const MAGIC_NEUR: [u8; 4] = *b"NEUR"; // 0x4E455552
pub const SECTOR_SIZE: usize = 512;
const ATTENTION_WEIGHT_BYTES: usize = 832;

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

/// Builds a 64 -> 32 -> 16 ternary MLP shard, padded to the requested LBA size.
pub fn build_shard_sector(version: u32, block_size: usize, layer1_weights: &[i8; 64]) -> Vec<u8> {
    assert!((512..=4096).contains(&block_size) && block_size.is_power_of_two());
    const INPUT_DIM: u32 = 64;
    const HIDDEN_DIM: u8 = 32;
    const OUTPUT_DIM: u16 = 16;
    const LAYER1_BYTES: usize = HIDDEN_DIM as usize * 16;
    const LAYER2_BYTES: usize = OUTPUT_DIM as usize * 8;
    let payload_size = 16 + LAYER1_BYTES + LAYER2_BYTES;
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
    sector[13..15].copy_from_slice(&OUTPUT_DIM.to_le_bytes());
    sector[15] = HIDDEN_DIM;

    // Layer 1: 32 rows of 64 ternary weights.
    let packed_layer1 = pack_weights_64(layer1_weights);
    let mut row = 0usize;
    while row < HIDDEN_DIM as usize {
        let start = 16 + row * packed_layer1.len();
        sector[start..start + packed_layer1.len()].copy_from_slice(&packed_layer1);
        row += 1;
    }

    // Layer 2: 16 rows of 32 positive ternary weights.
    let layer2_start = 16 + LAYER1_BYTES;
    let packed_positive = [0x55u8; 8];
    row = 0;
    while row < OUTPUT_DIM as usize {
        let start = layer2_start + row * packed_positive.len();
        sector[start..start + packed_positive.len()].copy_from_slice(&packed_positive);
        row += 1;
    }

    // The remainder of the final LBA block is zero-padded.
    sector
}

/// Builds a causal linear-attention shard with all-positive Q/K/V weights and
/// an identity O projection, padded to the requested LBA size.
pub fn build_attention_shard_sector(version: u32, block_size: usize) -> Vec<u8> {
    assert!((512..=4096).contains(&block_size) && block_size.is_power_of_two());
    let payload_size = 16 + ATTENTION_WEIGHT_BYTES;
    let padded_size = payload_size.div_ceil(block_size) * block_size;
    let mut sector = vec![0u8; padded_size];
    sector[0..4].copy_from_slice(&MAGIC_NEUR);
    sector[4..8].copy_from_slice(&version.to_le_bytes());
    sector[8..12].copy_from_slice(&64u32.to_le_bytes());
    sector[12] = 1; // Causal linear attention model type.
    sector[13..15].copy_from_slice(&16u16.to_le_bytes());
    sector[15] = 16; // Attention state dimension.

    // 00 01 01 01... repeated: every projection weight is +1.
    sector[16..16 + ATTENTION_WEIGHT_BYTES].fill(0x55);
    let output_start = 16 + (3 * 256);
    sector[output_start..output_start + 64].fill(0);
    for row in 0..16 {
        let row_start = output_start + row * 4;
        sector[row_start + (row >> 2)] |= 0x01 << ((row & 3) * 2);
    }
    sector
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out_path = "dist/production_shard.bin".to_string();
    let mut block_size = SECTOR_SIZE;
    let mut model_type = "mlp".to_string();
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

    const VERSION: u32 = 1;
    const INPUT_DIM: u32 = 64;
    const QUANT_TYPE: u8 = 0; // Ternary {-1, 0, 1}

    let sector = match model_type.as_str() {
        "mlp" => build_shard_sector(VERSION, block_size, &weights),
        "attention" => build_attention_shard_sector(VERSION, block_size),
        _ => return Err(format!("unsupported model type: {model_type} (use mlp or attention)").into()),
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
    println!("  - Input Dimension:    {} elements", INPUT_DIM);
    println!("  - Model Type:         {}", model_type);
    println!(
        "  - Hidden/Attention:   {} elements",
        if model_type == "attention" { 16 } else { 32 }
    );
    println!("  - Output Dimension:   16 elements");
    println!("  - LBA Block Size:     {} bytes", block_size);
    println!(
        "  - Quantization Type:  {} (Ternary {{-1, 0, 1}})",
        QUANT_TYPE
    );
    println!(
        "  - Payload Size:       {} bytes (LBA-block padded)",
        sector.len()
    );
    if model_type == "mlp" {
        println!("  - Layer 1 Size:       512 bytes");
        println!("  - Layer 2 Size:       128 bytes");
    } else {
        println!("  - Attention Size:     {} bytes", ATTENTION_WEIGHT_BYTES);
    }

    Ok(())
}
