# Standalone FAT16 & MBR appliance image builder (Zero dependencies)
param(
    [string]$efiPath = "c:\Users\IMOE001\9\target\x86_64-unknown-uefi\release\neural_box_core.efi",
    [string]$outPath = "c:\Users\IMOE001\9\scratch\test_custom_fat16.img"
)

$efiBytes = [System.IO.File]::ReadAllBytes($efiPath)
$efiSize = [uint32]$efiBytes.Length

$nshContent = "fs0:`r`n\EFI\BOOT\BOOTX64.EFI`r`n"
$nshBytes = [System.Text.Encoding]::ASCII.GetBytes($nshContent)
$nshSize = [uint32]$nshBytes.Length

# Partition Layout:
# LBA 0: MBR (512 bytes)
# LBA 2048: Partition 1 (ESP FAT16, 32768 sectors = 16 MiB)
# LBA 34816: Partition 2 (Raw data, 2048 sectors = 1 MiB)
# Total size: 36864 sectors = 18,874,368 bytes (18 MiB)

$totalSectors = 36864
$diskBytes = New-Object byte[] ($totalSectors * 512)

# --- 1. MBR (Sector 0) ---
$p1Offset = 446
$diskBytes[$p1Offset + 0] = 0x80 # Bootable
$diskBytes[$p1Offset + 1] = 0x00 # Head
$diskBytes[$p1Offset + 2] = 0x02 # Sector
$diskBytes[$p1Offset + 3] = 0x00 # Cyl
$diskBytes[$p1Offset + 4] = 0xEF # EFI System Partition
$diskBytes[$p1Offset + 5] = 0xFF
$diskBytes[$p1Offset + 6] = 0xFF
$diskBytes[$p1Offset + 7] = 0xFF
[BitConverter]::GetBytes([uint32]2048).CopyTo($diskBytes, $p1Offset + 8)
[BitConverter]::GetBytes([uint32]32768).CopyTo($diskBytes, $p1Offset + 12)

# Partition 2: Raw Block
$p2Offset = 446 + 16
$diskBytes[$p2Offset + 0] = 0x00
$diskBytes[$p2Offset + 4] = 0x83 # Linux / Raw
[BitConverter]::GetBytes([uint32]34816).CopyTo($diskBytes, $p2Offset + 8)
[BitConverter]::GetBytes([uint32]2048).CopyTo($diskBytes, $p2Offset + 12)

# MBR Signature
$diskBytes[510] = 0x55
$diskBytes[511] = 0xAA

# --- 2. FAT16 ESP Partition (starts at LBA 2048) ---
$part1ByteOffset = 2048 * 512

# BPB
$bpb = $part1ByteOffset
$diskBytes[$bpb + 0] = 0xEB; $diskBytes[$bpb + 1] = 0x3C; $diskBytes[$bpb + 2] = 0x90
[System.Text.Encoding]::ASCII.GetBytes("MSWIN4.1").CopyTo($diskBytes, $bpb + 3)
[BitConverter]::GetBytes([uint16]512).CopyTo($diskBytes, $bpb + 11)   # BytsPerSec
$diskBytes[$bpb + 13] = 4                                             # SecPerClus (2048 bytes per cluster)
[BitConverter]::GetBytes([uint16]4).CopyTo($diskBytes, $bpb + 14)     # RsvdSecCnt = 4
$diskBytes[$bpb + 16] = 2                                             # NumFATs = 2
[BitConverter]::GetBytes([uint16]512).CopyTo($diskBytes, $bpb + 17)   # RootEntCnt = 512
[BitConverter]::GetBytes([uint16]32768).CopyTo($diskBytes, $bpb + 19) # TotSec16 = 32768
$diskBytes[$bpb + 21] = 0xF8                                          # Media = 0xF8
[BitConverter]::GetBytes([uint16]64).CopyTo($diskBytes, $bpb + 22)    # FATSz16 = 64
[BitConverter]::GetBytes([uint16]63).CopyTo($diskBytes, $bpb + 24)    # SecPerTrk
[BitConverter]::GetBytes([uint16]255).CopyTo($diskBytes, $bpb + 26)   # NumHeads
[BitConverter]::GetBytes([uint32]2048).CopyTo($diskBytes, $bpb + 28)  # HiddSec
[BitConverter]::GetBytes([uint32]0).CopyTo($diskBytes, $bpb + 32)     # TotSec32 = 0
$diskBytes[$bpb + 36] = 0x80                                          # DrvNum
$diskBytes[$bpb + 38] = 0x29                                          # BootSig
[BitConverter]::GetBytes([uint32]0x12345678).CopyTo($diskBytes, $bpb + 39) # VolID
[System.Text.Encoding]::ASCII.GetBytes("EFI SYSTEM ").CopyTo($diskBytes, $bpb + 43)
[System.Text.Encoding]::ASCII.GetBytes("FAT16   ").CopyTo($diskBytes, $bpb + 54)
$diskBytes[$bpb + 510] = 0x55
$diskBytes[$bpb + 511] = 0xAA

# FAT1 at sector 4, FAT2 at sector 4 + 64 = 68
$fat1Offset = $part1ByteOffset + 4 * 512
$fat2Offset = $part1ByteOffset + 68 * 512

# Root dir at sector 4 + 2*64 = 132 (size 512 * 32 = 16384 bytes = 32 sectors)
$rootDirOffset = $part1ByteOffset + 132 * 512

# Data area starts at sector 132 + 32 = 164
# Cluster bytes = 4 * 512 = 2048
$dataAreaOffset = $part1ByteOffset + 164 * 512
function Get-ClusterOffset([int]$c) {
    return $dataAreaOffset + (($c - 2) * 2048)
}

# Directory entry helper
function Write-DirEntry([byte[]]$buf, [int]$offset, [string]$name11, [byte]$attr, [uint16]$startClus, [uint32]$size) {
    $bytes = [System.Text.Encoding]::ASCII.GetBytes($name11)
    [System.Array]::Copy($bytes, 0, $buf, $offset, 11)
    $buf[$offset + 11] = $attr
    [BitConverter]::GetBytes($startClus).CopyTo($buf, $offset + 26)
    [BitConverter]::GetBytes($size).CopyTo($buf, $offset + 28)
}

# Cluster allocation:
# Clus 2: directory "EFI"
# Clus 3: directory "BOOT"
# Clus 4..: file "BOOTX64.EFI"
$efiClustersNeeded = [int][Math]::Ceiling($efiSize / 2048.0)
if ($efiClustersNeeded -lt 1) { $efiClustersNeeded = 1 }
$efiStartClus = 4
$efiEndClus = $efiStartClus + $efiClustersNeeded - 1

# Clus ($efiEndClus + 1): file "STARTUP.NSH"
$nshClus = $efiEndClus + 1

# Populate Root Directory:
# Entry 1: "EFI        " (Dir, Clus 2)
# Entry 2: "STARTUP NSH" (File, Clus $nshClus)
Write-DirEntry $diskBytes ($rootDirOffset + 0)  "EFI        " 0x10 2 0
Write-DirEntry $diskBytes ($rootDirOffset + 32) "STARTUP NSH" 0x20 ([uint16]$nshClus) $nshSize

# Cluster 2 (EFI dir):
$clus2Offset = Get-ClusterOffset 2
Write-DirEntry $diskBytes ($clus2Offset + 0)  ".          " 0x10 2 0
Write-DirEntry $diskBytes ($clus2Offset + 32) "..         " 0x10 0 0
Write-DirEntry $diskBytes ($clus2Offset + 64) "BOOT       " 0x10 3 0

# Cluster 3 (BOOT dir):
$clus3Offset = Get-ClusterOffset 3
Write-DirEntry $diskBytes ($clus3Offset + 0)  ".          " 0x10 3 0
Write-DirEntry $diskBytes ($clus3Offset + 32) "..         " 0x10 2 0
Write-DirEntry $diskBytes ($clus3Offset + 64) "BOOTX64 EFI" 0x20 ([uint16]$efiStartClus) $efiSize

# Copy BOOTX64.EFI bytes to Cluster 4..
[System.Array]::Copy($efiBytes, 0, $diskBytes, (Get-ClusterOffset $efiStartClus), $efiBytes.Length)

# Copy startup.nsh bytes to Cluster $nshClus
[System.Array]::Copy($nshBytes, 0, $diskBytes, (Get-ClusterOffset $nshClus), $nshBytes.Length)

# Populate FAT Table (16-bit entries)
$fatEntries = New-Object uint16[] ($nshClus + 2)
$fatEntries[0] = 0xFFF8
$fatEntries[1] = 0xFFFF
$fatEntries[2] = 0xFFFF # Cluster 2 (EFI dir) end
$fatEntries[3] = 0xFFFF # Cluster 3 (BOOT dir) end

for ($k = 0; $k -lt $efiClustersNeeded; $k++) {
    $c = $efiStartClus + $k
    if ($k -eq ($efiClustersNeeded - 1)) {
        $fatEntries[$c] = 0xFFFF
    } else {
        $fatEntries[$c] = [uint16]($c + 1)
    }
}
$fatEntries[$nshClus] = 0xFFFF # startup.nsh end

for ($i = 0; $i -lt $fatEntries.Length; $i++) {
    $e = [BitConverter]::GetBytes($fatEntries[$i])
    [System.Array]::Copy($e, 0, $diskBytes, $fat1Offset + ($i * 2), 2)
    [System.Array]::Copy($e, 0, $diskBytes, $fat2Offset + ($i * 2), 2)
}

# --- 3. Partition 2 (Raw Payload Shard at LBA 34816) ---
$p2ByteOffset = 34816 * 512
# Magic: "NEUR"
$diskBytes[$p2ByteOffset + 0] = [byte][char]'N'
$diskBytes[$p2ByteOffset + 1] = [byte][char]'E'
$diskBytes[$p2ByteOffset + 2] = [byte][char]'U'
$diskBytes[$p2ByteOffset + 3] = [byte][char]'R'
[BitConverter]::GetBytes([uint32]1).CopyTo($diskBytes, $p2ByteOffset + 4)  # Version = 1
[BitConverter]::GetBytes([uint32]64).CopyTo($diskBytes, $p2ByteOffset + 8) # Dim = 64
$diskBytes[$p2ByteOffset + 12] = 0                                         # Quant = 0 (Ternary)

# Ingest weights from weights_shard.bin
$shardPath = "c:\Users\IMOE001\9\assets\weights_shard.bin"
if (Test-Path $shardPath) {
    $shardBytes = [System.IO.File]::ReadAllBytes($shardPath)
    [System.Array]::Copy($shardBytes, 0, $diskBytes, $p2ByteOffset + 16, 16)
}

[System.IO.File]::WriteAllBytes($outPath, $diskBytes)
Write-Host "Created custom FAT16 appliance image: $outPath ($($diskBytes.Length) bytes)"
