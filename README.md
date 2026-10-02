# Sovereign Neural Box

Sovereign Neural Box is a small, bare-metal x86-64 inference appliance. It boots as a UEFI application, loads a packed ternary model, and serves fixed-size input frames over a UART stream. The runtime combines two integer-only model paths: a feed-forward MLP and recurrent causal linear attention.

## Highlights

- **Dual-architecture inference:** 64→32→16 ternary MLP and 16-dimensional causal linear attention with a persistent 16×16 recurrent state.
- **No-heap inference core:** `src/main.rs` is `#![no_std]`; the core does not enable Rust `alloc` or use `Vec`/heap allocation. Model, DMA, frame, and state storage use fixed-size buffers. UEFI file I/O writes directly into the preallocated DMA-aligned shard buffer.
- **UEFI x86-64 target:** built for `x86_64-unknown-uefi` with the repository's nightly toolchain.
- **User-friendly model updates:** the GPT appliance image has a FAT32 EFI System Partition and a FAT32 `NEURAL_DATA` volume. Replace `NEURAL_DATA:\weights.bin` to load a different model at the next boot.
- **Fallback behavior:** UEFI SimpleFileSystem volumes are searched before `ExitBootServices`; invalid or missing files fall back to legacy raw-NVMe shard lookup, then to a safe built-in identity model.
- **UART streaming:** COM2 accepts 64 signed-byte inputs and returns 16 little-endian `i32` outputs. The standalone `NR` control marker resets recurrent attention state.

## Repository layout

- `src/` — UEFI entry point, model kernels, shard parsing, UART, NVMe, and shared-memory support.
- `tools/package_image.py` — GPT/FAT32 appliance image builder; `tools/package_image.ps1` is its PowerShell wrapper.
- `tools/payload_builder/` — host-side Rust NEUR shard generator (`mlp` or `attention`).
- `tools/test_dual_volume.ps1` — QEMU test for FAT-based model loading and UART streaming.
- `tools/test_attention_sequence.ps1` — QEMU test for recurrent attention accumulation and reset.
- `ml/` — optional model training and export utilities.
- `DEPLOYMENT.md` — detailed flashing, model-update, and server deployment guidance.

## Model format

A NEUR shard begins with a 16-byte header: ASCII magic `NEUR`, little-endian version and input dimension, a model-type byte, little-endian output dimension, and a final hidden/attention dimension byte. Ternary weights use two bits per weight: `00` is zero, `01` is +1, and `11` is −1.

| Model type | Value | Dimensions | Packed payload |
| --- | ---: | --- | ---: |
| Ternary MLP | `0` | 64 → 32 → 16 | 640 bytes |
| Causal linear attention | `1` | Q/K/V: 64 → 16; O: 16 → 16 | 832 bytes |

For attention, the recurrent state is a row-major 16×16 matrix of `i32` values. It accumulates key/value outer products across frames and is reset by the UART control marker.

## Build

Install the Rust nightly toolchain and the UEFI target listed in `rust-toolchain.toml`, then build the release EFI application:

```powershell
cargo +nightly build --target x86_64-unknown-uefi --release
```

The core is `no_std` and uses fixed storage for model weights, I/O frames, and attention state. File-system protocol metadata may be managed internally by UEFI firmware; no Rust heap allocator is enabled by this crate.

## Create the dual-volume image

Build the EFI binary first. The packager uses `dist/production_shard.bin` when present, or creates a small valid default MLP shard if it is absent.

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

The image can also be prepared with `tools/package_image.ps1`. See `DEPLOYMENT.md` before writing an image to physical media.

## UEFI model ingestion

Before leaving Boot Services, the application scans UEFI `SimpleFileSystem` handles using a fixed caller-owned handle array and searches for `\weights.bin` or `\NEURAL_WEIGHTS\weights.bin`. A valid shard is read directly into the 4 KiB DMA-aligned buffer and dispatched by its model-type field. If no file is found or parsing fails, the application attempts the legacy raw-NVMe locations; if that also fails, it runs the safe built-in fallback model.

To update a deployed appliance, mount the `NEURAL_DATA` FAT32 volume on a desktop OS and replace its root `weights.bin` with a valid NEUR shard. No EFI partition modification is needed.

## UART protocol

- **Input:** `NB` followed by exactly 64 raw signed `i8` bytes.
- **Output:** `NR`, one dimension byte, then that many little-endian `i32` values (16 for both supported models).
- **Attention reset:** send standalone `NR` with no following payload. This is a control event and does not produce an output frame.

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
