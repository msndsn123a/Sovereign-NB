# Test FAT32 image generation with MBR and Partition 2
param(
    [string]$efiPath = "c:\Users\IMOE001\9\target\x86_64-unknown-uefi\release\neural_box_core.efi",
    [string]$outPath = "c:\Users\IMOE001\9\scratch\test_appliance.img"
)

$efiBytes = [System.IO.File]::ReadAllBytes($efiPath)
$efiSize = [uint32]$efiBytes.Length

# Disk Layout:
# Sector size: 512 bytes
# LBA 0: MBR
# LBA 2048: Partition 1 (ESP FAT32, 65536 sectors = 32 MiB)
# LBA 67584: Partition 2 (Raw data, 2048 sectors = 1 MiB)
# Total size: 69632 sectors = 35,651,584 bytes

$totalSectors = 69632
$diskBytes = New-Object byte[] ($totalSectors * 512)

# --- 1. Master Boot Record (LBA 0) ---
# Partition 1: ESP (0xEF)
$p1Offset = 446
$diskBytes[$p1Offset + 0] = 0x80 # Bootable
$diskBytes[$p1Offset + 1] = 0x00 # Start Head
$diskBytes[$p1Offset + 2] = 0x02 # Start Sector
$diskBytes[$p1Offset + 3] = 0x00 # Start Cyl
$diskBytes[$p1Offset + 4] = 0xEF # EFI System Partition
$diskBytes[$p1Offset + 5] = 0xFF
$diskBytes[$p1Offset + 6] = 0xFF
$diskBytes[$p1Offset + 7] = 0xFF
# Start LBA: 2048
[BitConverter]::GetBytes([uint32]2048).CopyTo($diskBytes, $p1Offset + 8)
# Sectors: 65536 (32 MiB)
[BitConverter]::GetBytes([uint32]65536).CopyTo($diskBytes, $p1Offset + 12)

# Partition 2: Raw Block (0x83)
$p2Offset = 446 + 16
$diskBytes[$p2Offset + 0] = 0x00
$diskBytes[$p2Offset + 4] = 0x83 # Linux / Raw
# Start LBA: 67584
[BitConverter]::GetBytes([uint32]67584).CopyTo($diskBytes, $p2Offset + 8)
# Sectors: 2048 (1 MiB)
[BitConverter]::GetBytes([uint32]2048).CopyTo($diskBytes, $p2Offset + 12)

# Boot signature
$diskBytes[510] = 0x55
$diskBytes[511] = 0xAA

# --- 2. FAT32 Partition 1 (starts at byte offset 2048 * 512 = 1048576) ---
$part1ByteOffset = 2048 * 512

# BPB (Boot Sector at relative sector 0)
$bpb = $part1ByteOffset
$diskBytes[$bpb + 0] = 0xEB; $diskBytes[$bpb + 1] = 0x58; $diskBytes[$bpb + 2] = 0x90
[System.Text.Encoding]::ASCII.GetBytes("MSWIN4.1").CopyTo($diskBytes, $bpb + 3)
[BitConverter]::GetBytes([uint16]512).CopyTo($diskBytes, $bpb + 0x0B)   # BytsPerSec
$diskBytes[$bpb + 0x0D] = 8                                            # SecPerClus (4096 bytes)
[BitConverter]::GetBytes([uint16]32).CopyTo($diskBytes, $bpb + 0x0E)    # RsvdSecCnt
$diskBytes[$bpb + 0x10] = 2                                            # NumFATs
$diskBytes[$bpb + 0x15] = 0xF8                                         # Media
[BitConverter]::GetBytes([uint16]63).CopyTo($diskBytes, $bpb + 0x18)    # SecPerTrk
[BitConverter]::GetBytes([uint16]255).CopyTo($diskBytes, $bpb + 0x1A)   # NumHeads
[BitConverter]::GetBytes([uint32]2048).CopyTo($diskBytes, $bpb + 0x1C)  # HiddSec
[BitConverter]::GetBytes([uint32]65536).CopyTo($diskBytes, $bpb + 0x20) # TotSec32
[BitConverter]::GetBytes([uint32]64).CopyTo($diskBytes, $bpb + 0x24)    # FATSz32 (64 sectors per FAT)
[BitConverter]::GetBytes([uint32]2).CopyTo($diskBytes, $bpb + 0x2C)     # RootClus (Cluster 2)
[BitConverter]::GetBytes([uint16]1).CopyTo($diskBytes, $bpb + 0x30)     # FSInfo
[BitConverter]::GetBytes([uint16]6).CopyTo($diskBytes, $bpb + 0x32)     # BkBootSec
$diskBytes[$bpb + 0x40] = 0x80                                         # DrvNum
$diskBytes[$bpb + 0x42] = 0x29                                         # BootSig
[BitConverter]::GetBytes([uint32]0x12345678).CopyTo($diskBytes, $bpb + 0x43) # VolID
[System.Text.Encoding]::ASCII.GetBytes("EFI SYSTEM ").CopyTo($diskBytes, $bpb + 0x47)
[System.Text.Encoding]::ASCII.GetBytes("FAT32   ").CopyTo($diskBytes, $bpb + 0x52)
$diskBytes[$bpb + 510] = 0x55
$diskBytes[$bpb + 511] = 0xAA

# FSInfo Sector (relative sector 1)
$fsi = $part1ByteOffset + 512
[BitConverter]::GetBytes([uint32]0x41615252).CopyTo($diskBytes, $fsi + 0x00)
[BitConverter]::GetBytes([uint32]0x61417272).CopyTo($diskBytes, $fsi + 0x1E4)
[BitConverter]::GetBytes([uint32]8000).CopyTo($diskBytes, $fsi + 0x1E8)
[BitConverter]::GetBytes([uint32]6).CopyTo($diskBytes, $fsi + 0x1EC)
[BitConverter]::GetBytes([uint32]0xAA550000).CopyTo($diskBytes, $fsi + 0x1FC)

# Backup BPB & FSInfo (sectors 6 & 7)
[System.Array]::Copy($diskBytes, $bpb, $diskBytes, $part1ByteOffset + 6 * 512, 512)
[System.Array]::Copy($diskBytes, $fsi, $diskBytes, $part1ByteOffset + 7 * 512, 512)

# FAT Tables:
# FAT1 starts at sector 32
# FAT2 starts at sector 32 + 64 = 96
$fat1Offset = $part1ByteOffset + 32 * 512
$fat2Offset = $part1ByteOffset + 96 * 512

# Clusters:
# Clus 0: 0x0FFFFFF8
# Clus 1: 0x0FFFFFFF
# Clus 2 (Root Dir): 0x0FFFFFFF
# Clus 3 (EFI Dir):  0x0FFFFFFF
# Clus 4 (BOOT Dir): 0x0FFFFFFF
# Clus 5.. (BOOTX64.EFI file): chain ending with 0x0FFFFFFF
$clustersNeeded = [int][Math]::Ceiling($efiSize / 4096.0)
if ($clustersNeeded -lt 1) { $clustersNeeded = 1 }

$fatEntries = New-Object uint32[] (5 + $clustersNeeded)
$fatEntries[0] = 0x0FFFFFF8
$fatEntries[1] = 0x0FFFFFFF
$fatEntries[2] = 0x0FFFFFFF # Root dir
$fatEntries[3] = 0x0FFFFFFF # EFI dir
$fatEntries[4] = 0x0FFFFFFF # BOOT dir

for ($c = 0; $c -lt $clustersNeeded; $c++) {
    $clusIdx = 5 + $c
    if ($c -eq ($clustersNeeded - 1)) {
        $fatEntries[$clusIdx] = 0x0FFFFFFF # End of chain
    } else {
        $fatEntries[$clusIdx] = [uint32]($clusIdx + 1)
    }
}

for ($i = 0; $i -lt $fatEntries.Length; $i++) {
    $entryBytes = [BitConverter]::GetBytes($fatEntries[$i])
    [System.Array]::Copy($entryBytes, 0, $diskBytes, $fat1Offset + ($i * 4), 4)
    [System.Array]::Copy($entryBytes, 0, $diskBytes, $fat2Offset + ($i * 4), 4)
}

# Data Area starts at sector 32 + 2*64 = 160
# Cluster N byte offset = DataAreaStart + (N - 2) * ClusterBytes
$dataAreaOffset = $part1ByteOffset + 160 * 512

function Get-ClusterOffset([int]$clus) {
    return $dataAreaOffset + (($clus - 2) * 4096)
}

function Write-DirEntry([byte[]]$buf, [int]$offset, [string]$shortName, [byte]$attr, [int]$startClus, [uint32]$fileSize) {
    # 11-byte short name
    $nameBytes = [System.Text.Encoding]::ASCII.GetBytes($shortName.PadRight(11).Substring(0, 11).ToUpper())
    [System.Array]::Copy($nameBytes, 0, $buf, $offset, 11)
    $buf[$offset + 0x0B] = $attr
    [BitConverter]::GetBytes([uint16](($startClus -shr 16) -band 0xFFFF)).CopyTo($buf, $offset + 0x14)
    [BitConverter]::GetBytes([uint16]($startClus -band 0xFFFF)).CopyTo($buf, $offset + 0x1A)
    [BitConverter]::GetBytes($fileSize).CopyTo($buf, $offset + 0x1C)
}

# Cluster 2: Root Directory contains entry for "EFI"
$rootOffset = Get-ClusterOffset 2
Write-DirEntry $diskBytes $rootOffset "EFI" 0x10 3 0

# Cluster 3: EFI Directory contains entries for ".", "..", and "BOOT"
$efiOffset = Get-ClusterOffset 3
Write-DirEntry $diskBytes ($efiOffset + 0)  ".          " 0x10 3 0
Write-DirEntry $diskBytes ($efiOffset + 32) "..         " 0x10 0 0
Write-DirEntry $diskBytes ($efiOffset + 64) "BOOT" 0x10 4 0

# Cluster 4: BOOT Directory contains entries for ".", "..", and "BOOTX64.EFI"
$bootOffset = Get-ClusterOffset 4
Write-DirEntry $diskBytes ($bootOffset + 0)  ".          " 0x10 4 0
Write-DirEntry $diskBytes ($bootOffset + 32) "..         " 0x10 3 0
Write-DirEntry $diskBytes ($bootOffset + 64) "BOOTX64 EFI" 0x20 5 $efiSize

# Cluster 5+: BOOTX64.EFI file data
$fileDataOffset = Get-ClusterOffset 5
[System.Array]::Copy($efiBytes, 0, $diskBytes, $fileDataOffset, $efiBytes.Length)

# --- 3. Partition 2 (Raw Shard at LBA 67584 = byte offset 67584 * 512 = 34603008) ---
$part2Offset = 67584 * 512

# Create production shard header:
# Magic: "NEUR" (0x4E455552)
$diskBytes[$part2Offset + 0] = [byte][char]'N'
$diskBytes[$part2Offset + 1] = [byte][char]'E'
$diskBytes[$part2Offset + 2] = [byte][char]'U'
$diskBytes[$part2Offset + 3] = [byte][char]'R'
[BitConverter]::GetBytes([uint32]1).CopyTo($diskBytes, $part2Offset + 4)  # Version = 1
[BitConverter]::GetBytes([uint32]64).CopyTo($diskBytes, $part2Offset + 8) # Dim = 64
$diskBytes[$part2Offset + 12] = 0                                        # Quant = 0 (Ternary)

# Ingest packed weights at offset 16 (same as weights_shard.bin)
$shardPath = "c:\Users\IMOE001\9\assets\weights_shard.bin"
if (Test-Path $shardPath) {
    $shardBytes = [System.IO.File]::ReadAllBytes($shardPath)
    [System.Array]::Copy($shardBytes, 0, $diskBytes, $part2Offset + 16, 16)
}

[System.IO.File]::WriteAllBytes($outPath, $diskBytes)
Write-Host "Appliance image generated successfully: $outPath ($($diskBytes.Length) bytes)"
