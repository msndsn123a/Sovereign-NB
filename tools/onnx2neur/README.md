# onnx2neur

A dependency-light Rust CLI that reads ONNX `ModelProto`/`GraphProto` protobuf data, extracts constant float32 linear weights, quantizes them, and writes an authenticated NEUR v2 artifact. Only Gemm/MatMul-only graphs are accepted; variable weights, nonzero Gemm biases, non-float32 initializers, and other operators fail closed.

## Build and run

```powershell
cargo +nightly build --manifest-path tools/onnx2neur/Cargo.toml --target x86_64-pc-windows-msvc --release
cargo +nightly run --manifest-path tools/onnx2neur/Cargo.toml --target x86_64-pc-windows-msvc --release -- --input model.onnx --output dist/model.neur --key-file signing.seed --quant ternary
```

The key file is a raw 32-byte Ed25519 seed and is required; the public key is printed with the output details. Keep the seed private and do not commit it. Ternary export supports a connected 64→32→16 two-layer MLP and emits the appliance's packed signed ternary layout. `--quant pot` supports four attention projections (three 16×64 Q/K/V matrices and one 16×16 O matrix) and emits signed-nibble PoT weights. `--threshold` defaults to `0.25`; PoT exponents are rounded to the nearest power of two and clamped to 2⁶. `--pot-scale` accepts 0–6 for the attention fixed-point right shift. `--block-size` accepts powers of two from 512 through 4096.

The roundtrip unit test constructs an exported reference `ModelProto`, converts it, checks NEUR v2 fields and padding, and verifies the Ed25519 signature over the exact UEFI-authenticated message.
