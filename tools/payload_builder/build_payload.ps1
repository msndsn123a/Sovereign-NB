# PowerShell Raw Shard Builder Tool
param(
    [string]$outPath = "dist/production_shard.bin",
    [ValidateRange(512, 4096)][int]$blockSize = 512,
    [ValidateSet("mlp", "attention")][string]$model = "mlp"
)

if (($blockSize -band ($blockSize - 1)) -ne 0) {
    throw "blockSize must be a power of two"
}

$outDir = Split-Path -Parent $outPath
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$inputDim = 64
$hiddenDim = if ($model -eq "attention") { 16 } else { 32 }
$outputDim = 16
$payloadSize = if ($model -eq "attention") { 16 + 832 } else { 16 + ($hiddenDim * 16) + ($outputDim * 8) }
$sectorSize = [int][Math]::Ceiling($payloadSize / $blockSize) * $blockSize
$sector = New-Object byte[] $sectorSize

# 1. Magic bytes: "NEUR" (0x4E455552)
$sector[0] = [byte][char]'N'
$sector[1] = [byte][char]'E'
$sector[2] = [byte][char]'U'
$sector[3] = [byte][char]'R'

# 2. Version = 1
[BitConverter]::GetBytes([uint32]1).CopyTo($sector, 4)

# 3. Input Dim = 64
[BitConverter]::GetBytes([uint32]64).CopyTo($sector, 8)

# 4. Quant Type = 0 (Ternary)
$sector[12] = if ($model -eq "attention") { 1 } else { 0 }
[BitConverter]::GetBytes([uint16]$outputDim).CopyTo($sector, 13)
$sector[15] = [byte]$hiddenDim

if ($model -eq "attention") {
    # Q/K/V all-positive weights create 4096 in state[0] per [1; 64] frame.
    # O is identity so emitted values also make the changing recurrent state visible.
    for ($index = 16; $index -lt (16 + 832); $index++) { $sector[$index] = 0x55 }
    $outputStart = 16 + (3 * 256)
    for ($index = $outputStart; $index -lt ($outputStart + 64); $index++) { $sector[$index] = 0 }
    for ($row = 0; $row -lt 16; $row++) {
        $byteIndex = $outputStart + ($row * 4) + ($row -shr 2)
        $sector[$byteIndex] = [byte]($sector[$byteIndex] -bor (0x01 -shl (($row -band 3) * 2)))
    }
    [System.IO.File]::WriteAllBytes($outPath, $sector)
    Write-Host "[PAYLOAD_BUILDER]: Produced $outPath ($sectorSize bytes, model=attention, dims=64->16, block_size=$blockSize)"
    return
}

# 5. Pack 64 weights
$weightPattern = @(1, -1, 0, 1)
$weights = @(0..63)
for ($w = 0; $w -lt 64; $w++) {
    $weights[$w] = $weightPattern[($w + ($w -shr 2)) % 4]
}

function Encode-Ternary([int]$val) {
    if ($val -eq 1) { return 0x01 }
    if ($val -eq -1) { return 0x03 }
    return 0x00
}

for ($row = 0; $row -lt $hiddenDim; $row++) {
    for ($b = 0; $b -lt 16; $b++) {
        $base = $b * 4
        $b0 = Encode-Ternary $weights[$base]
        $b1 = (Encode-Ternary $weights[$base + 1]) -shl 2
        $b2 = (Encode-Ternary $weights[$base + 2]) -shl 4
        $b3 = (Encode-Ternary $weights[$base + 3]) -shl 6
        $sector[16 + ($row * 16) + $b] = [byte]($b0 -bor $b1 -bor $b2 -bor $b3)
    }
}

# Layer 2: 16 rows of 32 positive ternary weights (8 bytes per row).
$layer2Start = 16 + ($hiddenDim * 16)
for ($row = 0; $row -lt $outputDim; $row++) {
    for ($b = 0; $b -lt 8; $b++) {
        $sector[$layer2Start + ($row * 8) + $b] = 0x55
    }
}

[System.IO.File]::WriteAllBytes($outPath, $sector)
Write-Host "[PAYLOAD_BUILDER]: Produced $outPath ($sectorSize bytes, dims=$inputDim->$hiddenDim->$outputDim, block_size=$blockSize)"
