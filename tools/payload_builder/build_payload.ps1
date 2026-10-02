param(
    [string]$outPath = "dist/production_shard.bin",
    [ValidateRange(512, 4096)][int]$blockSize = 512,
    [ValidateSet("mlp", "attention")][string]$model = "mlp",
    [ValidateSet("pattern", "zero")][string]$variant = "pattern",
    [ValidateSet("ternary", "pot")][string]$quant = "ternary",
    [ValidateRange(0, 6)][int]$potScale = 0,
    [switch]$multiStream,
    [switch]$pruneBlocks,
    [switch]$activationLut,
    [string]$keyFile = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repoRoot
try {
    $arguments = @("tools/payload_builder/build_payload.py", $outPath, "--block-size", "$blockSize", "--model", $model, "--variant", $variant, "--quant", $quant, "--pot-scale", "$potScale")
    if ($multiStream) { $arguments += "--multi-stream" }
    if ($pruneBlocks) { $arguments += "--prune-blocks" }
    if ($activationLut) { $arguments += "--activation-lut" }
    if ($keyFile) { $arguments += @("--key-file", $keyFile) }
    & python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Signed shard generation failed: $LASTEXITCODE" }
} finally {
    Pop-Location
}
