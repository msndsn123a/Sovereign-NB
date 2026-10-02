#!/usr/bin/env python3
"""Python entry point for the canonical Ed25519-signed NEUR v2 Rust builder."""

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="dist/production_shard.bin")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--model", choices=("mlp", "attention"), default="mlp")
    parser.add_argument("--variant", choices=("pattern", "zero"), default="pattern")
    parser.add_argument("--quant", choices=("ternary", "pot"), default="ternary")
    parser.add_argument("--multi-stream", action="store_true")
    parser.add_argument("--pot-scale", type=int, choices=range(0, 7), default=0)
    parser.add_argument("--prune-blocks", action="store_true")
    parser.add_argument("--activation-lut", action="store_true")
    parser.add_argument("--key-file", help="optional raw 32-byte Ed25519 seed file")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rustc = subprocess.run(
        ["rustc", "-vV"], check=True, capture_output=True, text=True, cwd=root
    ).stdout
    host = next(line.split(":", 1)[1].strip() for line in rustc.splitlines() if line.startswith("host:"))
    command = [
        "cargo", "run", "--manifest-path", "tools/payload_builder/Cargo.toml",
        "--target", host, "--release", "--", "--model", args.model,
        "--block-size", str(args.block_size), "--output", args.output,
        "--variant", args.variant,
        "--quant", args.quant, "--pot-scale", str(args.pot_scale),
    ]
    if args.multi_stream:
        command.append("--multi-stream")
    if args.prune_blocks:
        command.append("--prune-blocks")
    if args.activation_lut:
        command.append("--activation-lut")
    if args.key_file:
        command.extend(("--key-file", args.key_file))
    subprocess.run(command, check=True, cwd=root)


if __name__ == "__main__":
    main()
