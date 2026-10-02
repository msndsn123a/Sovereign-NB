param(
    [string]$efiPath = "target/x86_64-unknown-uefi/release/neural_box_core.efi",
    [string]$shardPath = "dist/production_shard.bin",
    [string]$outPath = "dist/neural_box_appliance.img"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    $python = Get-Command python -ErrorAction Stop
    & $python.Source tools/package_image.py --efi-path $efiPath --shard-path $shardPath --output $outPath
    if ($LASTEXITCODE -ne 0) { throw "Dual-volume image packaging failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}
