#!/usr/bin/env python3
"""Build a GPT appliance image with an ESP and user-accessible NEURAL_DATA FAT32 volume."""

import argparse
import math
import os
import struct
import subprocess
import uuid
import zlib

SECTOR_SIZE = 512
PARTITION_SECTORS = 131_072  # 64 MiB; enough clusters for a standards-compliant FAT32 volume.
ESP_START_LBA = 2048
DATA_START_LBA = ESP_START_LBA + PARTITION_SECTORS
TOTAL_SECTORS = DATA_START_LBA + PARTITION_SECTORS + 2048
HOT_SWAP_LBA = DATA_START_LBA + PARTITION_SECTORS
FAT32_RESERVED_SECTORS = 32
FAT_COUNT = 2


def generate_signed_default_shard(shard_path):
    """Generate a signed default shard with the Rust test-key signer."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rustc = subprocess.run(
        ["rustc", "-vV"], check=True, capture_output=True, text=True, cwd=repo_root
    ).stdout
    host = next(line.split(":", 1)[1].strip() for line in rustc.splitlines() if line.startswith("host:"))
    command = [
        "cargo", "run", "--manifest-path", "tools/payload_builder/Cargo.toml",
        "--target", host, "--release", "--", "--model", "mlp", "--output", shard_path,
    ]
    os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
    subprocess.run(command, check=True, cwd=repo_root)


def fat32_format(disk, start_lba, sector_count, volume_label, volume_id, files, esp=False):
    """Format one FAT32 partition directly into the output disk bytearray."""
    sectors_per_cluster = 1
    fat_sectors = 1
    while True:
        data_sectors = sector_count - FAT32_RESERVED_SECTORS - FAT_COUNT * fat_sectors
        cluster_count = data_sectors // sectors_per_cluster
        needed_fat_sectors = math.ceil((cluster_count + 2) * 4 / SECTOR_SIZE)
        # The equation can oscillate by one sector at boundary cluster counts.
        # Keeping the larger result is valid and guarantees monotonic convergence.
        adjusted_fat_sectors = max(fat_sectors, needed_fat_sectors)
        if adjusted_fat_sectors == fat_sectors:
            break
        fat_sectors = adjusted_fat_sectors

    if cluster_count < 65_525:
        raise ValueError("FAT32 volume is too small for the FAT32 cluster-count minimum")

    volume_start = start_lba * SECTOR_SIZE
    data_start_sector = FAT32_RESERVED_SECTORS + FAT_COUNT * fat_sectors
    data_start = volume_start + data_start_sector * SECTOR_SIZE
    fat = bytearray(fat_sectors * SECTOR_SIZE)
    next_cluster = 2

    def reserve_chain(payload):
        nonlocal next_cluster
        count = max(1, math.ceil(len(payload) / SECTOR_SIZE))
        first = next_cluster
        for offset in range(count):
            cluster = next_cluster
            next_cluster += 1
            following = cluster + 1 if offset + 1 < count else 0x0FFFFFFF
            struct.pack_into("<I", fat, cluster * 4, following)
            chunk = payload[offset * SECTOR_SIZE:(offset + 1) * SECTOR_SIZE]
            byte_offset = data_start + (cluster - 2) * SECTOR_SIZE
            disk[byte_offset:byte_offset + len(chunk)] = chunk
        return first, len(payload)

    def short_entry(name, attributes, first_cluster=0, file_size=0):
        entry = bytearray(32)
        encoded = name.encode("ascii")
        if len(encoded) != 11:
            raise ValueError(f"FAT short name must be 11 bytes: {name!r}")
        entry[:11] = encoded
        entry[11] = attributes
        struct.pack_into("<H", entry, 20, (first_cluster >> 16) & 0xFFFF)
        struct.pack_into("<H", entry, 26, first_cluster & 0xFFFF)
        struct.pack_into("<I", entry, 28, file_size)
        return entry

    def write_directory(cluster, entries):
        payload = b"".join(entries) + bytes(SECTOR_SIZE - len(entries) * 32)
        byte_offset = data_start + (cluster - 2) * SECTOR_SIZE
        disk[byte_offset:byte_offset + SECTOR_SIZE] = payload

    # Reserved FAT entries and root directory cluster.
    struct.pack_into("<I", fat, 0, 0x0FFFFFF8)
    struct.pack_into("<I", fat, 4, 0x0FFFFFFF)
    struct.pack_into("<I", fat, 8, 0x0FFFFFFF)
    next_cluster = 3  # Cluster 2 is reserved for the root directory.

    if esp:
        root_cluster = 2
        efi_cluster = 3
        boot_cluster = 4
        struct.pack_into("<I", fat, efi_cluster * 4, 0x0FFFFFFF)
        struct.pack_into("<I", fat, boot_cluster * 4, 0x0FFFFFFF)
        next_cluster = 5

        efi_entry = short_entry("EFI        ", 0x10, efi_cluster)
        write_directory(root_cluster, [efi_entry])
        write_directory(efi_cluster, [
            short_entry(".          ", 0x10, efi_cluster),
            short_entry("..         ", 0x10, root_cluster),
            short_entry("BOOT       ", 0x10, boot_cluster),
        ])
        efi_bytes = files["BOOTX64.EFI"]
        efi_first, efi_size = reserve_chain(efi_bytes)
        write_directory(boot_cluster, [
            short_entry(".          ", 0x10, boot_cluster),
            short_entry("..         ", 0x10, efi_cluster),
            short_entry("BOOTX64 EFI", 0x20, efi_first, efi_size),
        ])
        if "STARTUP.NSH" in files:
            nsh_first, nsh_size = reserve_chain(files["STARTUP.NSH"])
            # A small root-directory entry list is sufficient for this appliance.
            write_directory(root_cluster, [efi_entry, short_entry("STARTUP NSH", 0x20, nsh_first, nsh_size)])
    else:
        root_cluster = 2
        label = volume_label.encode("ascii")[:11].ljust(11, b" ")
        entries = [short_entry(label.decode("ascii"), 0x08, root_cluster)]
        for name, payload in files.items():
            if name != "weights.bin":
                continue
            first_cluster, file_size = reserve_chain(payload)
            entries.append(short_entry("WEIGHTS BIN", 0x20, first_cluster, file_size))
        write_directory(root_cluster, entries)

    # Write both FAT copies after all chains have been allocated.
    fat_start = volume_start + FAT32_RESERVED_SECTORS * SECTOR_SIZE
    fat_bytes = fat_sectors * SECTOR_SIZE
    disk[fat_start:fat_start + fat_bytes] = fat
    second_fat = fat_start + fat_bytes
    disk[second_fat:second_fat + fat_bytes] = fat

    # FAT32 BPB / boot sector.
    boot = bytearray(SECTOR_SIZE)
    boot[0:3] = b"\xEB\x58\x90"
    boot[3:11] = b"MSWIN4.1"
    struct.pack_into("<H", boot, 11, SECTOR_SIZE)
    boot[13] = sectors_per_cluster
    struct.pack_into("<H", boot, 14, FAT32_RESERVED_SECTORS)
    boot[16] = FAT_COUNT
    struct.pack_into("<H", boot, 17, 0)  # FAT32 has no fixed root-entry table.
    struct.pack_into("<H", boot, 19, 0)
    boot[21] = 0xF8
    struct.pack_into("<H", boot, 22, 0)
    struct.pack_into("<HHI", boot, 24, 63, 255, start_lba)
    struct.pack_into("<I", boot, 32, sector_count)
    struct.pack_into("<I", boot, 36, fat_sectors)
    struct.pack_into("<HHIHH", boot, 40, 0, 0, root_cluster, 1, 6)
    boot[64] = 0x80
    boot[66] = 0x29
    struct.pack_into("<I", boot, 67, volume_id)
    boot[71:82] = volume_label.encode("ascii")[:11].ljust(11, b" ")
    boot[82:90] = b"FAT32   "
    boot[510:512] = b"\x55\xAA"
    disk[volume_start:volume_start + SECTOR_SIZE] = boot
    backup_boot = volume_start + 6 * SECTOR_SIZE
    disk[backup_boot:backup_boot + SECTOR_SIZE] = boot

    fsinfo = bytearray(SECTOR_SIZE)
    struct.pack_into("<I", fsinfo, 0, 0x41615252)
    struct.pack_into("<I", fsinfo, 484, 0x61417272)
    struct.pack_into("<I", fsinfo, 488, max(0, cluster_count - (next_cluster - 2)))
    struct.pack_into("<I", fsinfo, 492, next_cluster)
    fsinfo[510:512] = b"\x55\xAA"
    fsinfo_start = volume_start + SECTOR_SIZE
    disk[fsinfo_start:fsinfo_start + SECTOR_SIZE] = fsinfo
    backup_fsinfo = volume_start + 7 * SECTOR_SIZE
    disk[backup_fsinfo:backup_fsinfo + SECTOR_SIZE] = fsinfo


def gpt_entry(type_guid, name, first_lba, last_lba):
    entry = bytearray(128)
    entry[:16] = uuid.UUID(type_guid).bytes_le
    entry[16:32] = uuid.uuid4().bytes_le
    struct.pack_into("<QQQ", entry, 32, first_lba, last_lba, 0)
    encoded_name = name.encode("utf-16le")[:72]
    entry[56:56 + len(encoded_name)] = encoded_name
    return entry


def write_gpt(disk):
    total_sectors = len(disk) // SECTOR_SIZE
    last_lba = total_sectors - 1
    first_usable = 34
    last_usable = total_sectors - 34
    entry_array = bytearray(128 * 128)
    entry_array[0:128] = gpt_entry(
        "C12A7328-F81F-11D2-BA4B-00A0C93EC93B",
        "EFI System", ESP_START_LBA, ESP_START_LBA + PARTITION_SECTORS - 1,
    )
    entry_array[128:256] = gpt_entry(
        "EBD0A0A2-B9E5-4433-87C0-68B6B72699C7",
        "NEURAL_DATA", DATA_START_LBA, DATA_START_LBA + PARTITION_SECTORS - 1,
    )
    entries_crc = zlib.crc32(entry_array) & 0xFFFFFFFF
    primary_entries_lba = 2
    backup_entries_lba = last_lba - 32
    disk[primary_entries_lba * SECTOR_SIZE:(primary_entries_lba + 32) * SECTOR_SIZE] = entry_array
    disk[backup_entries_lba * SECTOR_SIZE:(backup_entries_lba + 32) * SECTOR_SIZE] = entry_array

    disk_guid = uuid.uuid4().bytes_le

    def header(current_lba, backup_lba, entries_lba):
        raw = bytearray(SECTOR_SIZE)
        struct.pack_into(
            "<8sIIIIQQQQ16sQIII", raw, 0,
            b"EFI PART", 0x00010000, 92, 0, 0,
            current_lba, backup_lba, first_usable, last_usable,
            disk_guid, entries_lba, 128, 128, entries_crc,
        )
        struct.pack_into("<I", raw, 16, zlib.crc32(raw[:92]) & 0xFFFFFFFF)
        return raw

    disk[SECTOR_SIZE:2 * SECTOR_SIZE] = header(1, last_lba, primary_entries_lba)
    disk[last_lba * SECTOR_SIZE:(last_lba + 1) * SECTOR_SIZE] = header(
        last_lba, 1, backup_entries_lba
    )

    # Protective MBR, with a single 0xEE partition spanning the GPT disk.
    mbr = bytearray(SECTOR_SIZE)
    mbr[446] = 0
    mbr[450] = 0xEE
    struct.pack_into("<II", mbr, 454, 1, min(last_lba, 0xFFFFFFFF))
    mbr[510:512] = b"\x55\xAA"
    disk[:SECTOR_SIZE] = mbr


def package_appliance_image(
    efi_path="target/x86_64-unknown-uefi/release/neural_box_core.efi",
    shard_path="dist/production_shard.bin",
    out_path="dist/neural_box_appliance.img",
    hot_swap_shard_path=None,
):
    if not os.path.exists(efi_path):
        fallback_efi = "esp/EFI/BOOT/BOOTX64.EFI"
        if os.path.exists(fallback_efi):
            efi_path = fallback_efi
        else:
            raise FileNotFoundError(f"EFI binary not found at {efi_path}")
    with open(efi_path, "rb") as file:
        efi_bytes = file.read()

    shard_bytes = b""
    if os.path.exists(shard_path):
        with open(shard_path, "rb") as file:
            shard_bytes = file.read()
    if (
        len(shard_bytes) < 80
        or shard_bytes[:4] != b"NEUR"
        or struct.unpack_from("<I", shard_bytes, 4)[0] != 2
    ):
        if os.path.normcase(os.path.normpath(shard_path)) != os.path.normcase(
            os.path.normpath("dist/production_shard.bin")
        ):
            raise ValueError("custom shard is unsigned/legacy; provide a signed NEUR v2 shard")
        print("[PACKAGE_IMAGE]: Legacy/unsigned default shard found; generating signed NEUR v2 default")
        generate_signed_default_shard(shard_path)
        with open(shard_path, "rb") as file:
            shard_bytes = file.read()
    print(f"[PACKAGE_IMAGE]: Using signed NEUR v2 shard {shard_path} ({len(shard_bytes)} bytes)")

    disk = bytearray(TOTAL_SECTORS * SECTOR_SIZE)
    startup_script = b"fs0:\r\n\\EFI\\BOOT\\BOOTX64.EFI\r\n"
    fat32_format(
        disk, ESP_START_LBA, PARTITION_SECTORS, "EFI SYSTEM", 0x12345678,
        {"BOOTX64.EFI": efi_bytes, "STARTUP.NSH": startup_script}, esp=True,
    )
    fat32_format(
        disk, DATA_START_LBA, PARTITION_SECTORS, "NEURAL_DATA", 0x4E455552,
        {"weights.bin": shard_bytes},
    )
    write_gpt(disk)

    if hot_swap_shard_path:
        with open(hot_swap_shard_path, "rb") as file:
            update_shard = file.read()
        backup_entries_lba = TOTAL_SECTORS - 1 - 32
        available_bytes = (backup_entries_lba - HOT_SWAP_LBA) * SECTOR_SIZE
        if len(update_shard) > available_bytes:
            raise ValueError("hot-swap shard exceeds the reserved raw-NVMe region")
        hot_swap_offset = HOT_SWAP_LBA * SECTOR_SIZE
        disk[hot_swap_offset:hot_swap_offset + len(update_shard)] = update_shard
        print(
            f"[PACKAGE_IMAGE]: Hot-swap shard {hot_swap_shard_path} written at raw LBA {HOT_SWAP_LBA} ({len(update_shard)} bytes)"
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as file:
        file.write(disk)
    print(f"[PACKAGE_IMAGE]: Created {out_path} ({len(disk)} bytes)")
    print(
        "[PACKAGE_IMAGE]: GPT partitions: ESP FAT32 "
        f"LBA {ESP_START_LBA}-{ESP_START_LBA + PARTITION_SECTORS - 1}; "
        "NEURAL_DATA FAT32 "
        f"LBA {DATA_START_LBA}-{DATA_START_LBA + PARTITION_SECTORS - 1}"
    )
    print(f"[PACKAGE_IMAGE]: NEURAL_DATA\\weights.bin populated ({len(shard_bytes)} bytes)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--efi-path", default="target/x86_64-unknown-uefi/release/neural_box_core.efi")
    parser.add_argument("--shard-path", default="dist/production_shard.bin")
    parser.add_argument("--output", default="dist/neural_box_appliance.img")
    parser.add_argument("--hot-swap-shard", help="signed shard to place in the reserved raw-NVMe update region")
    args = parser.parse_args()
    package_appliance_image(args.efi_path, args.shard_path, args.output, args.hot_swap_shard)


if __name__ == "__main__":
    main()
