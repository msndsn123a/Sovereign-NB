param(
    [string]$Python = "C:/Users/IMOE001/python311/python.exe",
    [int]$Port = 5558,
    [string]$QemuAccel = "whpx",
    [int]$BlockSize = 512
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
try {
    if (-not (Test-Path $Python)) {
        throw "Python executable not found: $Python"
    }

    & $Python ml/train_pure_mlp.py --checkpoint dist/pure_mlp_qat.pt --metrics dist/pure_mlp_qat_metrics.json
    if ($LASTEXITCODE -ne 0) { throw "PyTorch QAT training failed with exit code $LASTEXITCODE" }

    & $Python ml/export_pure_shard.py --checkpoint dist/pure_mlp_qat.pt --output dist/trained_pure_mlp_shard.bin --expected dist/trained_pure_mlp_expected.json --block-size $BlockSize
    if ($LASTEXITCODE -ne 0) { throw "NEUR shard export failed with exit code $LASTEXITCODE" }

    cargo +nightly build --target x86_64-unknown-uefi --release
    if ($LASTEXITCODE -ne 0) { throw "UEFI release build failed with exit code $LASTEXITCODE" }

    & .\tools\test_uart_roundtrip.ps1 -Port $Port -QemuAccel $QemuAccel -ShardPath dist/trained_pure_mlp_shard.bin -ExpectedPath dist/trained_pure_mlp_expected.json
    if ($LASTEXITCODE -ne 0) { throw "QEMU trained-shard verification failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
