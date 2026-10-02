#!/usr/bin/env python3
"""Export a trained QAT checkpoint into a validated NEUR 64->32->16 shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import struct
from pathlib import Path

import torch

from train_pure_mlp import (
    HIDDEN_DIM,
    INPUT_DIM,
    OUTPUT_DIM,
    PureTernaryMLP,
    ternary_quantize,
)

MAGIC = b"NEUR"
UNSIGNED_INPUT_VERSION = 1
VERSION = 2
QUANT_TERNARY = 0
HEADER_SIZE = 16
LAYER1_BYTES = 32 * 16
LAYER2_BYTES = 16 * 8
PAYLOAD_SIZE = HEADER_SIZE + LAYER1_BYTES + LAYER2_BYTES


def quantized_int8(tensor: torch.Tensor) -> list[list[int]]:
    return ternary_quantize(tensor.detach().cpu()).to(torch.int8).tolist()


def pack_row(weights: list[int], expected_count: int) -> bytes:
    if len(weights) != expected_count:
        raise ValueError(f"weight row has {len(weights)} values; expected {expected_count}")
    packed = bytearray((expected_count + 3) // 4)
    encoding = {0: 0b00, 1: 0b01, -1: 0b11}
    for index, weight in enumerate(weights):
        if weight not in encoding:
            raise ValueError(f"non-ternary quantized weight: {weight}")
        packed[index // 4] |= encoding[weight] << ((index % 4) * 2)
    return bytes(packed)


def validate_packed(data: bytes) -> None:
    if any(((byte >> shift) & 0b11) == 0b10 for byte in data for shift in (0, 2, 4, 6)):
        raise ValueError("packed weight payload uses invalid ternary code 10b")


def encode_shard(layer1: list[list[int]], layer2: list[list[int]], block_size: int) -> bytes:
    if len(layer1) != HIDDEN_DIM or any(len(row) != INPUT_DIM for row in layer1):
        raise ValueError("layer1 dimensions must be exactly (32, 64)")
    if len(layer2) != OUTPUT_DIM or any(len(row) != HIDDEN_DIM for row in layer2):
        raise ValueError("layer2 dimensions must be exactly (16, 32)")
    if block_size not in (512, 4096):
        raise ValueError("block_size must be either 512 or 4096")

    payload = bytearray()
    for row in layer1:
        payload.extend(pack_row(row, INPUT_DIM))
    for row in layer2:
        payload.extend(pack_row(row, HIDDEN_DIM))
    if len(payload) != LAYER1_BYTES + LAYER2_BYTES:
        raise ValueError("internal packed payload size mismatch")
    validate_packed(payload)

    header = bytearray(HEADER_SIZE)
    header[0:4] = MAGIC
    struct.pack_into("<I", header, 4, UNSIGNED_INPUT_VERSION)
    struct.pack_into("<I", header, 8, INPUT_DIM)
    header[12] = QUANT_TERNARY
    struct.pack_into("<H", header, 13, OUTPUT_DIM)
    header[15] = HIDDEN_DIM

    shard = header + payload
    shard.extend(bytes((-len(shard)) % block_size))
    validate_header_and_size(bytes(shard), block_size)
    return bytes(shard)


def validate_header_and_size(shard: bytes, block_size: int) -> None:
    if len(shard) % block_size != 0 or len(shard) < PAYLOAD_SIZE:
        raise ValueError("shard size is not LBA-padded or is truncated")
    if shard[0:4] != MAGIC:
        raise ValueError("bad NEUR magic")
    version, input_dim = struct.unpack_from("<II", shard, 4)
    quant_type = shard[12]
    output_dim = struct.unpack_from("<H", shard, 13)[0]
    hidden_dim = shard[15]
    if (version, input_dim, hidden_dim, output_dim, quant_type) != (
        UNSIGNED_INPUT_VERSION, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, QUANT_TERNARY
    ):
        raise ValueError("NEUR header fields do not match the 64->32->16 ternary format")
    validate_packed(shard[HEADER_SIZE:PAYLOAD_SIZE])


def run_integer_reference(inputs: list[int], layer1: list[list[int]], layer2: list[list[int]]) -> list[int]:
    if len(inputs) != INPUT_DIM:
        raise ValueError(f"test vector must contain exactly {INPUT_DIM} signed bytes")
    if any(value < -128 or value > 127 for value in inputs):
        raise ValueError("test vector values must fit signed int8")
    hidden = []
    for row in layer1:
        accumulator = sum(int(value) * weight for value, weight in zip(inputs, row))
        hidden.append(1 if accumulator > 0 else (-1 if accumulator < 0 else 0))
    return [sum(value * weight for value, weight in zip(hidden, row)) for row in layer2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="dist/pure_mlp_qat.pt")
    parser.add_argument("--output", default="dist/trained_pure_mlp_shard.bin")
    parser.add_argument("--expected", default="dist/trained_pure_mlp_expected.json")
    parser.add_argument("--block-size", type=int, default=512, choices=(512, 4096))
    parser.add_argument("--version", type=int, default=VERSION)
    parser.add_argument("--key-file", help="optional raw 32-byte Ed25519 seed file")
    args = parser.parse_args()
    if args.version != VERSION:
        raise ValueError("this exporter emits signed NEUR version 2")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model = PureTernaryMLP()
    model.load_state_dict(state_dict, strict=True)

    layer1 = quantized_int8(model.layer1.weight)
    layer2 = quantized_int8(model.layer2.weight)
    unsigned_shard = encode_shard(layer1, layer2, args.block_size)

    # Deterministic signed-int8 batch, serialized for frame-by-frame QEMU comparison.
    test_batch = [
        [((index * 7 + batch_index * 11) % 9) - 4 for index in range(INPUT_DIM)]
        for batch_index in range(8)
    ]
    integer_reference = [run_integer_reference(vector, layer1, layer2) for vector in test_batch]
    model.eval()
    with torch.no_grad():
        pytorch_output = model(torch.tensor(test_batch, dtype=torch.float32))
        expected_batch = pytorch_output.to(torch.int32).tolist()
    if expected_batch != integer_reference:
        raise ValueError(
            "quantized PyTorch forward pass differs from exact integer ternary reference: "
            f"torch={expected_batch}, integer={integer_reference}"
        )
    output_path = Path(args.output)
    expected_path = Path(args.expected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    unsigned_path = output_path.with_suffix(output_path.suffix + ".unsigned")
    unsigned_path.write_bytes(unsigned_shard)
    repo_root = Path(__file__).resolve().parent.parent
    rustc = subprocess.run(
        ["rustc", "-vV"], check=True, capture_output=True, text=True, cwd=repo_root
    ).stdout
    host = next(line.split(":", 1)[1].strip() for line in rustc.splitlines() if line.startswith("host:"))
    sign_command = [
        "cargo", "run", "--manifest-path", "tools/payload_builder/Cargo.toml",
        "--target", host, "--release", "--", "--unsigned-input", str(unsigned_path),
        "--output", str(output_path), "--block-size", str(args.block_size),
    ]
    if args.key_file:
        sign_command.extend(("--key-file", args.key_file))
    try:
        subprocess.run(sign_command, check=True, cwd=repo_root)
    finally:
        unsigned_path.unlink(missing_ok=True)
    shard = output_path.read_bytes()
    if len(shard) % args.block_size or len(shard) < 80 + LAYER1_BYTES + LAYER2_BYTES:
        raise ValueError("signed NEUR v2 shard has invalid size")
    signed_version = struct.unpack_from("<I", shard, 4)[0]
    if shard[:4] != MAGIC or signed_version != 2:
        raise ValueError("signer did not produce a NEUR v2 shard")
    expected = {
        "test_batch_i8": test_batch,
        "test_batch_output_i32": expected_batch,
        "dimensions": {"input": INPUT_DIM, "hidden": HIDDEN_DIM, "output": OUTPUT_DIM},
        "shard_sha256": hashlib.sha256(shard).hexdigest(),
    }
    expected_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")

    print(
        "[SHARD EXPORT]: magic=NEUR version=2 input_dim=64 hidden_dim=32 output_dim=16 quant_type=0 signature=Ed25519"
    )
    print(f"[SHARD EXPORT]: layer1={LAYER1_BYTES} bytes layer2={LAYER2_BYTES} bytes")
    print(f"[SHARD EXPORT]: unsigned_payload={PAYLOAD_SIZE - HEADER_SIZE} bytes signed_header=80 bytes padded_size={len(shard)} block_size={args.block_size}")
    print(f"[SHARD EXPORT]: sha256={expected['shard_sha256']}")
    print(f"[SHARD EXPORT]: shard={output_path} expected={expected_path}")
    print(
        f"[SHARD EXPORT]: PyTorch/integer reference parity=match for {len(test_batch)} test vectors"
    )
    print(f"[SHARD EXPORT]: first QEMU reference output={expected_batch[0]}")


if __name__ == "__main__":
    main()
