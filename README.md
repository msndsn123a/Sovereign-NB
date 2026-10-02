[![Live Wasm Playground](https://img.shields.io/badge/Live%20Demo-Wasm%20In--Browser%20Playground-brightgreen)](https://msndsn123a.github.io/Sovereign-NB/)

# Sovereign Neural Box

Sovereign Neural Box is a `no_std` bare-metal UEFI inference project. Its x86-64 appliance boots a packed ternary model and serves fixed-size input frames over UART; an AArch64 UEFI target exercises the NEON integer kernel and firmware-preserving boot path. Both model math and the in-browser verifier use exact integer operations.

## Highlights

- **Dual-architecture inference:** 64→32→16 ternary MLP and 16-dimensional causal linear attention with a persistent 16×16 recurrent state.
- **No-heap inference core:** `src/main.rs` is `#![no_std]`; the core does not enable Rust `alloc` or use `Vec`/heap allocation. Model, DMA, frame, and state storage use fixed-size buffers. UEFI file I/O writes directly into the preallocated DMA-aligned shard buffer.
- **UEFI x86-64 target:** built for `x86_64-unknown-uefi` with the repository's nightly toolchain.
- **AArch64 UEFI target:** cross-compiles for `aarch64-unknown-uefi` and includes an exact-parity NEON MLP path.
- **User-friendly model updates:** the GPT appliance image has a FAT32 EFI System Partition and a FAT32 `NEURAL_DATA` volume. Replace `NEURAL_DATA:\weights.bin` to load a different model at the next boot.
- **Fallback behavior:** UEFI SimpleFileSystem volumes are searched before `ExitBootServices`; missing files may use legacy raw-NVMe lookup, while invalid files are rejected to the safe built-in identity model.
- **Signed weights:** NEUR v2 shards authenticate their metadata and payload with Ed25519 before `ExitBootServices`; invalid or unsigned files are rejected to the safe identity model.
- **UART streaming:** COM2 accepts 64 signed-byte inputs and returns 16 little-endian `i32` outputs. The standalone `NR` control marker resets recurrent attention state.

## 20 bare-metal engineering milestones

1. Fixed-shape 64→32→16 ternary MLP inference.
2. AVX-512 integer ternary acceleration.
3. AVX-512 VPOPCNTDQ row-parallel acceleration.
4. Bit-plane/scalar/vector exact-output parity checks.
5. Recurrent causal linear attention with a 16×16 state matrix.
6. Independent attention state for eight streams.
7. Integer SiLU lookup table for fixed-point non-linearities.
8. Structured 8-input-block pruning with active masks.
9. Ed25519-authenticated NEUR v2 model shards.
10. Safe identity-model fallback for invalid or missing shards.
11. Polled-mode NVMe queues for bounded model reads.
12. Shadow-buffer, atomic model hot-swapping without stopping ingress.
13. PCIe MMIO/IVSHMEM inference ingress.
14. Uncached page mappings for MMIO ranges.
15. Custom CR3 identity maps with 1 GiB/2 MiB huge pages.
16. Intel L3 CAT cache-way isolation where supported.
17. ACPI-described SMP discovery and xAPIC SIPI AP startup.
18. CPUID-gated UMWAIT with bounded PAUSE fallback.
19. Intel TME / AMD SME probing, strict encryption policy, and AMD C-bit tagging.
20. AArch64 bare-metal UEFI cross-compilation with NEON kernel validation.

## Interactive Wasm playground

Open the [live in-browser verifier](https://msndsn123a.github.io/Sovereign-NB/). It runs the no-std Rust kernels in WebAssembly, randomizes signed-byte inputs, displays all 16 outputs, compares them element-by-element with the scalar reference, and visualizes per-stream recurrent attention state. It has no external CDN or JavaScript dependencies.

The repository's **Settings → Pages → Build and deployment → Source** must be set to **GitHub Actions** for the workflow to publish updates.

To run locally, build the Wasm artifact and serve the standalone page:

```powershell
.\tools\wasm_verifier\build.ps1
node tools/wasm_verifier/serve.js
```

Then open `http://127.0.0.1:8000/`.

## Repository layout

- `src/` — UEFI entry point, model kernels, shard parsing, UART, NVMe, and shared-memory support.
- `tools/package_image.py` — GPT/FAT32 appliance image builder; `tools/package_image.ps1` is its PowerShell wrapper.
- `tools/payload_builder/` — host-side Rust NEUR shard generator (`mlp` or `attention`).
- `tools/onnx2neur/` — native Rust ONNX-to-NEUR v2 converter with Ed25519 signing.
- `tools/wasm_verifier/` — shared integer kernels, browser parity playground, and Pages deployment source.
- `.github/workflows/deploy-pages.yml` — builds and deploys the Wasm playground to GitHub Pages.
- `tools/test_dual_volume.ps1` — QEMU test for FAT-based model loading and UART streaming.
- `tools/test_attention_sequence.ps1` — QEMU test for recurrent attention accumulation and reset.
- `ml/` — optional model training and export utilities.
- `DEPLOYMENT.md` — detailed flashing, model-update, and server deployment guidance.

## Model format

A signed NEUR v2 shard begins with an 80-byte header: the 16-byte metadata prefix (ASCII magic `NEUR`, little-endian version and input dimension, model type, output dimension and hidden/attention dimension) followed by a 64-byte Ed25519 signature. The signature covers the metadata prefix concatenated with the packed model payload. Ternary weights use two bits per weight: `00` is zero, `01` is +1, and `11` is −1.

| Model type | Value | Dimensions | Packed payload |
| --- | ---: | --- | ---: |
| Ternary MLP | `0` | 64 → 32 → 16 | 640 bytes |
| Causal linear attention | `1` | Q/K/V: 64 → 16; O: 16 → 16 | 832 bytes |

For attention, the recurrent state is a row-major 16×16 matrix of `i32` values. It accumulates key/value outer products across frames and is reset by the UART control marker.

| Model type | Value | Dimensions | Packed payload | Signed shard minimum |
| --- | ---: | --- | ---: | ---: |
| Ternary MLP | `0` | 64 → 32 → 16 | 640 bytes | 720 bytes |
| Causal linear attention | `1` | Q/K/V: 64 → 16; O: 16 → 16 | 832 bytes | 912 bytes |

## Build

Install the Rust nightly toolchain and the UEFI target listed in `rust-toolchain.toml`, then build the release EFI application:

```powershell
cargo +nightly build --target x86_64-unknown-uefi --release
```

The core is `no_std` and uses fixed storage for model weights, I/O frames, and attention state. File-system protocol metadata may be managed internally by UEFI firmware; no Rust heap allocator is enabled by this crate.

## Create the dual-volume image

Build the EFI binary first. The packager uses a signed NEUR v2 shard from `dist/production_shard.bin`; if that default path contains a legacy unsigned shard or is missing, it invokes the Rust builder to generate a signed test MLP shard.

```powershell
python tools/package_image.py
```

This creates `dist/neural_box_appliance.img`, a GPT disk image with a protective MBR:

1. **ESP:** FAT32, contains `\EFI\BOOT\BOOTX64.EFI` and `\STARTUP.NSH`.
2. **NEURAL_DATA:** FAT32, contains the default model as `\weights.bin`.

For an Attention shard, generate it with the host builder and pass it as the packager's shard input (the default packaging path is `dist/production_shard.bin`):

```powershell
cargo run --manifest-path tools/payload_builder/Cargo.toml --target x86_64-pc-windows-msvc --release -- --model attention --output dist/production_shard.bin
python tools/package_image.py
```

The host builder accepts `--key-file <path>` for a raw 32-byte Ed25519 seed. If omitted, it uses a deterministic RFC 8032 development/test key whose public half is embedded in the firmware; do not use that known test key for production deployments. Production releases should replace the firmware public key and sign shards with a separately protected private key.

The image can also be prepared with `tools/package_image.ps1`. See `DEPLOYMENT.md` before writing an image to physical media.

## UEFI model ingestion

Before leaving Boot Services, the application scans UEFI `SimpleFileSystem` handles using a fixed caller-owned handle array and searches for `\weights.bin` or `\NEURAL_WEIGHTS\weights.bin`. A file is verified in place in the 4 KiB DMA-aligned buffer before `ExitBootServices` and dispatched by its model-type field only after strict Ed25519 verification. An invalid or unsigned file is rejected and falls back directly to the safe identity model; legacy raw-NVMe lookup is attempted only when no file is found.

To update a deployed appliance, mount the `NEURAL_DATA` FAT32 volume on a desktop OS and replace its root `weights.bin` with a valid NEUR shard. No EFI partition modification is needed.

## UART protocol

- **Input (legacy stream 0):** `NB` followed by exactly 64 raw signed `i8` bytes.
- **Input (selected stream):** `NS`, one stream ID byte (`0..7`), then exactly 64 raw signed `i8` bytes. IDs above 7 are rejected. Nonzero IDs require the signed model metadata to enable multi-stream mode.
- **Output:** `NR`, one dimension byte, then that many little-endian `i32` values (16 for both supported models).
- **Selective attention reset:** `NR` followed by one stream ID byte (`0..7`) resets only that stream and continues the session. A lone `NR` remains the legacy stream-0 reset-and-exit control and produces no output frame.
- **Shadow shard update:** `NU` followed by an 8-byte little-endian raw NVMe LBA queues an update on the auxiliary AP. A candidate is read into the inactive DMA buffer and Ed25519-verified before the active model index is swapped; failed reads/signatures leave the active model unchanged. The BSP continues accepting frames while the AP updates.

The attention payload builder accepts `--multi-stream` to sign the eight-stream capability and `--quant pot --pot-scale <0..6>` to emit packed signed PoT weights. PoT nibbles encode zero, `+2^k` (1..7), or `-2^k` (9..15); nibble 8 is reserved. The scale and capability flags are included in the authenticated NEUR metadata.

For MLP shards, `--prune-blocks` appends row-wise active masks for 8-input blocks in both layers; masked-off blocks must contain zero weights and are skipped by scalar and AVX-512 sparse kernels. `--activation-lut` selects the cache-line-aligned, fixed-point SiLU lookup table for the hidden activation. Both settings are authenticated in the NEUR v2 metadata/payload.

The UEFI streaming loop has a finite frame limit and timeout intended for appliance/QEMU verification. COM1 carries text diagnostics; COM2 carries the binary protocol.

## QEMU verification

With QEMU installed and `assets/OVMF.fd` available:

```powershell
cargo +nightly build --target x86_64-unknown-uefi --release
python tools/package_image.py
.\tools\test_dual_volume.ps1
```

The dual-volume test boots the GPT image with the disk attached as NVMe, sends eight input frames, verifies 16-element responses, and asserts that COM1 reports a FAT-loaded shard and zero dropped frames. To verify attention state and reset:

```powershell
.\tools\test_attention_sequence.ps1
```

## License

Licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
