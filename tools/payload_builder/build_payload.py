#!/usr/bin/env python3
"""
Offline Raw Shard Builder Tool for Neural-Box Core.
Packages arbitrary model weights into the exact hardware-ingestible raw sector format.
"""

import struct
import os

MAGIC_NEUR = b"NEUR"  # 0x4E455552
SECTOR_SIZE = 512

def encode_ternary(w: int) -> int:
    if w == 1:
        return 0b01
    elif w == -1:
        return 0b11
    else:
        return 0b00

def pack_weights_64(weights: list[int]) -> bytes:
    assert len(weights) == 64
    packed = bytearray(16)
    for b in range(16):
        base = b * 4
        b0 = encode_ternary(weights[base])
        b1 = encode_ternary(weights[base + 1]) << 2
        b2 = encode_ternary(weights[base + 2]) << 4
        b3 = encode_ternary(weights[base + 3]) << 6
        packed[b] = b0 | b1 | b2 | b3
    return bytes(packed)

def build_shard_sector(
    version: int = 1,
    weights: list[int] = None,
    block_size: int = SECTOR_SIZE,
) -> bytes:
    if not 512 <= block_size <= 4096 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two from 512 through 4096")
    if weights is None:
        pattern = [1, -1, 0, 1]
        weights = [pattern[(w + (w >> 2)) & 3] for w in range(64)]

    input_dim = 64
    hidden_dim = 32
    output_dim = 16
    layer1_bytes = hidden_dim * 16
    layer2_bytes = output_dim * 8
    payload_size = 16 + layer1_bytes + layer2_bytes
    padded_size = ((payload_size + block_size - 1) // block_size) * block_size
    sector = bytearray(padded_size)
    # Magic bytes (0..4): "NEUR"
    sector[0:4] = MAGIC_NEUR
    # Version (4..8, uint32 little-endian)
    struct.pack_into("<I", sector, 4, version)
    # Input Dim (8..12, uint32 little-endian)
    struct.pack_into("<I", sector, 8, input_dim)
    # Quant Type (12, uint8)
    sector[12] = 0
    # Output dimension is little-endian at 13..15; hidden dimension occupies byte 15.
    struct.pack_into("<H", sector, 13, output_dim)
    sector[15] = hidden_dim
    # Layer 1: row-major 64-input weights; this fixture repeats the sample row.
    packed = pack_weights_64(weights)
    for row in range(hidden_dim):
        start = 16 + row * len(packed)
        sector[start:start + len(packed)] = packed
    # Layer 2: 16 rows, each containing 32 positive ternary weights.
    layer2_start = 16 + layer1_bytes
    packed_positive = bytes([0x55] * 8)
    for row in range(output_dim):
        start = layer2_start + row * len(packed_positive)
        sector[start:start + len(packed_positive)] = packed_positive
    return bytes(sector)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="dist/production_shard.bin")
    parser.add_argument("--block-size", type=int, default=SECTOR_SIZE)
    args = parser.parse_args()
    out_path = args.output

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = build_shard_sector(block_size=args.block_size)
    with open(out_path, "wb") as f:
        f.write(payload)

    print(f"[PAYLOAD_BUILDER]: Produced {out_path} ({len(payload)} bytes, 64->32->16)")

if __name__ == "__main__":
    main()
