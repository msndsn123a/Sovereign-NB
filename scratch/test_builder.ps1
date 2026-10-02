# Build test appliance with qemu-img generated FAT + Partition 2
param(
    [string]$outPath = "c:\Users\IMOE001\9\scratch\test_appliance2.img"
)

# 1. Convert esp directory to FAT16 disk image
$tempImg = "c:\Users\IMOE001\9\scratch\temp_fat.img"
& "C:\Users\IMOE001\AppData\Local\Programs\qemu\qemu-img.exe" convert -f raw -O raw "fat:16:esp" $tempImg

# 2. Read first 65536 sectors (32 MiB) of the FAT image
$fs = [System.IO.File]::OpenRead($tempImg)
$part1Bytes = New-Object byte[] (65536 * 512)
$read = $fs.Read($part1Bytes, 0, $part1Bytes.Length)
$fs.Close()
Remove-Item $tempImg -Force -ErrorAction SilentlyContinue

# Total sectors: 65536 (P1) + 2048 (P2) = 67584 sectors (33 MiB)
$totalBytes = New-Object byte[] (67584 * 512)
[System.Array]::Copy($part1Bytes, 0, $totalBytes, 0, $part1Bytes.Length)

# 3. Update MBR Partition Table in Sector 0
# Partition 1: Start LBA 63, Size: (65536 - 63) = 65473 sectors
$p1Offset = 446
$totalBytes[$p1Offset + 4] = 0xEF # EFI System Partition
[BitConverter]::GetBytes([uint32]63).CopyTo($totalBytes, $p1Offset + 8)
[BitConverter]::GetBytes([uint32]65473).CopyTo($totalBytes, $p1Offset + 12)

# Update BPB of Partition 1 at LBA 63
$bpbOffset = 63 * 512
[BitConverter]::GetBytes([uint16]65473).CopyTo($totalBytes, $bpbOffset + 19) # TotSec16
[BitConverter]::GetBytes([uint32]0).CopyTo($totalBytes, $bpbOffset + 32)     # TotSec32 = 0

# Partition 2: Start LBA 65536, Size: 2048 sectors (1 MiB)
$p2Offset = 446 + 16
$totalBytes[$p2Offset + 0] = 0x00
$totalBytes[$p2Offset + 4] = 0x83 # Linux / Raw
[BitConverter]::GetBytes([uint32]65536).CopyTo($totalBytes, $p2Offset + 8)
[BitConverter]::GetBytes([uint32]2048).CopyTo($totalBytes, $p2Offset + 12)

# 4. Write production shard at LBA 65536 (Partition 2 start)
$p2ByteOffset = 65536 * 512
# Magic: "NEUR"
$totalBytes[$p2ByteOffset + 0] = [byte][char]'N'
$totalBytes[$p2ByteOffset + 1] = [byte][char]'E'
$totalBytes[$p2ByteOffset + 2] = [byte][char]'U'
$totalBytes[$p2ByteOffset + 3] = [byte][char]'R'
[BitConverter]::GetBytes([uint32]1).CopyTo($totalBytes, $p2ByteOffset + 4)  # Version = 1
[BitConverter]::GetBytes([uint32]64).CopyTo($totalBytes, $p2ByteOffset + 8) # Dim = 64
$totalBytes[$p2ByteOffset + 12] = 0                                         # Quant = 0 (Ternary)

# Ingest weights from weights_shard.bin
$shardPath = "c:\Users\IMOE001\9\assets\weights_shard.bin"
if (Test-Path $shardPath) {
    $shardBytes = [System.IO.File]::ReadAllBytes($shardPath)
    [System.Array]::Copy($shardBytes, 0, $totalBytes, $p2ByteOffset + 16, 16)
}

[System.IO.File]::WriteAllBytes($outPath, $totalBytes)
Write-Host "Created test appliance image: $outPath ($($totalBytes.Length) bytes)"
