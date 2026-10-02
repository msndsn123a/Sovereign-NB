# NEURAL-BOX CORE: BARE-METAL APPLIANCE DEPLOYMENT GUIDE

`neural_box_appliance.img` is a self-contained, bootable bare-metal appliance that boots directly into pure-logic CPU execution without an operating system, runtime, or hypervisor.

---

## 1. Appliance Architecture & Block Layout

The appliance image is a GPT disk with a protective MBR and two standard FAT32
partitions. The model volume is user-accessible as `NEURAL_DATA` on desktop systems.

```text
+---------------------------------------------------------------------------------+
| Sector 0: Protective MBR (GPT)                                                 |
+---------------------------------------------------------------------------------+
| Primary GPT header + partition entries; 1 MiB alignment headroom                |
+---------------------------------------------------------------------------------+
| Partition 1 (LBA 2048 .. 133119, 64 MiB): EFI System Partition (FAT32)          |
|   ├── /EFI/BOOT/BOOTX64.EFI   (Sovereign Core UEFI Binary)                      |
|   └── /STARTUP.NSH             (UEFI Shell launch helper)                       |
+---------------------------------------------------------------------------------+
| Partition 2 (LBA 133120 .. 264191, 64 MiB): NEURAL_DATA (FAT32)                 |
|   └── /weights.bin             (Default valid NEUR model shard)                  |
+---------------------------------------------------------------------------------+
| Backup GPT entries + header                                                      |
+---------------------------------------------------------------------------------+
```

At boot, the UEFI core searches SimpleFileSystem volumes for `\weights.bin` or
`\NEURAL_WEIGHTS\weights.bin` before `ExitBootServices`. Copy a replacement model
to the root of the mounted `NEURAL_DATA` volume as `weights.bin`; raw NVMe LBA
loading remains a fallback for legacy images.

---

## 2. Local Flashing: Physical USB Drive

### Windows (Rufus)
1. Insert a USB flash drive (at least 160 MiB capacity; 256 MiB or larger recommended).
2. Launch **Rufus** (v3.x or v4.x).
3. Under **Device**, select your target USB thumb drive.
4. Under **Boot selection**, choose `dist/neural_box_appliance.img`.
5. Keep the image's GPT partition table; select **UEFI (non-CSM)**.
6. When prompted, select **Write in DD Image mode**.
7. Click **START** to write the block image bit-for-bit to the drive.

### Linux / macOS (`dd`)
Identify your target disk device (e.g., `/dev/sdb` on Linux or `/dev/rdisk2` on macOS):
```bash
# Verify the device node carefully to avoid overwriting host data!
lsblk

# Flash the raw image to the thumb drive
sudo dd if=dist/neural_box_appliance.img of=/dev/sdX bs=4M status=progress conv=fsync
```

---

## 3. Remote Server Deployment: IPMI / iDRAC / iLO

### Dell PowerEdge (iDRAC 8 / 9)
1. Log into the iDRAC Web Interface.
2. Navigate to **Configuration** → **Virtual Media** → **Virtual Media Management**.
3. Under **Attach Virtual Media**, select `dist/neural_box_appliance.img` as **Virtual Floppy / Removable Drive** or **Virtual Optical Drive**.
4. In **Boot Settings**, set Next Boot Device to **Virtual Media / Removable**.
5. Open the **Virtual Console** and power on the chassis.
6. Verify output over Serial-Over-LAN (SOL) or the virtual console.

### HPE ProLiant (iLO 4 / 5 / 6)
1. Open the iLO Integrated Remote Console (IRC).
2. From the top menu, select **Virtual Drives** → **Image File**.
3. Select `dist/neural_box_appliance.img` and check **Connect**.
4. Reboot the server and select **One-Time Boot Menu (F11)** → select the attached Virtual Drive.

### Supermicro / Generic AMI MegaRAC BMC
1. Access the BMC web interface and open the HTML5 / Java KVM.
2. Select **Virtual Media** → **CD-ROM Image** or **Floppy Image**.
3. Mount `dist/neural_box_appliance.img`.
4. Power cycle the system via IPMI:
   ```bash
   ipmitool -H <BMC_IP> -U <USER> -P <PASS> power reset
   ```

---

## 4. Bare-Metal NVMe Flashing (Direct-to-Silicon)

For high-throughput edge nodes, flash the appliance directly to the onboard NVMe storage from a Linux rescue / live shell:

### Complete Appliance Flashing (Bootloader + Model Weights)
```bash
# Write the unified appliance to primary NVMe disk
sudo dd if=dist/neural_box_appliance.img of=/dev/nvme0n1 bs=4M status=progress conv=fsync
```

### Drag-and-Drop Model Updates (Zero Bootloader Disturbance)
Mount the `NEURAL_DATA` FAT32 partition and replace `weights.bin` with a valid NEUR
shard. For example, on Linux:
```bash
# Mount the user-accessible model volume and copy the replacement shard
sudo mkdir -p /mnt/NEURAL_DATA
sudo mount /dev/nvme0n1p2 /mnt/NEURAL_DATA
sudo cp dist/production_shard.bin /mnt/NEURAL_DATA/weights.bin
sync
sudo umount /mnt/NEURAL_DATA
```

---

## 5. BIOS / UEFI Hardware Configuration Checklist

Before booting the appliance, ensure the target motherboard firmware is configured as follows:

| Setting | Required Value | Notes |
| :--- | :--- | :--- |
| **Boot Mode** | `UEFI Only` | Disable Legacy Boot / CSM (Compatibility Support Module) |
| **Secure Boot** | `Disabled` | Required for custom sovereign UEFI execution |
| **Serial Port (COM1)** | `Enabled` | Base address `0x3F8`, IRQ 4, Baud rate 115200 (8-N-1) |
| **Console Redirection** | `COM1 / Serial-Over-LAN` | Enables real-time telemetry ingestion via BMC or RS-232 |
| **C-States / P-States** | `Disabled / Maximum Performance` | Guarantees zero latency spikes and fixed CPU clock |
| **Hyper-Threading** | `Disabled` (Optional) | Recommended for deterministic single-core sovereignty |

### Hardware Memory Encryption Policy (TME / SME)

At startup, COM1 reports Intel TME or AMD SME only after the CPU vendor and
CPUID capability bits advertise the corresponding MSRs. Unsupported or
virtualized CPUs report `[SECURITY MEM]: Hardware encryption unsupported /
inactive (virtualized/legacy)` without attempting an unsupported `RDMSR`.
AMD SME's CPUID-provided C-bit is applied to non-MMIO identity-map huge pages;
MMIO mappings remain unencrypted and uncached. Intel TME is transparent to the
page-table format.

Mission-critical builds can make hardware encryption mandatory with the
compile-time Cargo policy feature:

```powershell
cargo +nightly build --features strict-memory-encryption --target x86_64-unknown-uefi --release
```

Strict mode refuses boot before model loading or DMA setup unless Intel TME is
active, locked, and reports a recognized AES-XTS algorithm, or AMD SME is active
with a valid C-bit position. The default build reports status but allows
virtualized/legacy hardware for development and QEMU regression tests.

Expected security records include:

```text
[SECURITY TME]: Intel TME active (AES-XTS-128, locked=true)
[SECURITY SME]: AMD SME active (encrypted DRAM, C-bit=47, SEV supported=false)
[SECURITY MEM]: Hardware encryption unsupported / inactive (virtualized/legacy)
```

---

## 6. Remote Telemetry Capture (Serial-Over-LAN)

To ingest cycle-accurate telemetry remotely over IPMI Serial-Over-LAN:
```bash
ipmitool -H <BMC_IP> -U <USER> -P <PASS> sol activate
```

Expected COM1 telemetry stream:
```text
[SOVEREIGN_CORE]: SERIAL TELEMETRY INITIALIZED (COM1 115200 8-N-1)
NEURAL-BOX CORE: HARNESS INITIALIZED
BITNET KERNEL: VERIFIED 100% MATCH [Scalar == AVX512]
[SOVEREIGN_CORE]: BOOT SERVICES TERMINATED. INTERRUPTS MUTED. CPU ACQUIRED.
[SOVEREIGN_CORE]: NVMe DETECTED & INITIALIZED VIA MMIO.
[SOVEREIGN_CORE]: PRODUCTION SHARD HEADER DETECTED AT LBA 34816 (MAGIC: NEUR).
[SOVEREIGN_CORE]: SHARD METADATA: VERSION=1, DIM=64, QUANT=0
[SOVEREIGN_CORE]: DMA STREAM INGESTED (LBA -> MEMORY ZERO-COPY).
[SOVEREIGN_CORE]: INFERENCE ON INGESTED WEIGHTS VERIFIED.
[SOVEREIGN_CORE]: 100K ITERATIONS COMPLETE. ZERO JITTER.
[SOVEREIGN_CORE]: STREAMING INFERENCE ACTIVE (50K FRAMES).
[SOVEREIGN_CORE]: LATENCY METRICS: MIN = 10108 CYCLES, AVG = 14572 CYCLES, MAX = 3269094 CYCLES.
[SOVEREIGN_CORE]: ESTIMATED LATENCY: ~5828 NS PER INFERENCE PASS (@ 2500 MHz).
[SOVEREIGN_CORE]: DETERMINISTIC REAL-TIME CRITERIA: SATISFIED (SUB-MICROSECOND).
```
