$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$cargoToml = Join-Path $PSScriptRoot "Cargo.toml"
cargo +nightly build --manifest-path $cargoToml --no-default-features --target wasm32-unknown-unknown --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$artifact = Join-Path $PSScriptRoot "target\wasm32-unknown-unknown\release\wasm_verifier.wasm"
$destination = Join-Path $PSScriptRoot "wasm_verifier.wasm"
Copy-Item -LiteralPath $artifact -Destination $destination -Force
Write-Host "[WASM VERIFIER]: built $destination ($((Get-Item $destination).Length) bytes)"
