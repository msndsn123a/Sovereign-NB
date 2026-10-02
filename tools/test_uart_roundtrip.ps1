param(
    [int]$Port = 5556,
    [string]$QemuAccel = "",
    [string]$QemuCpu = "Skylake-Server,+avx512f,+avx512dq",
    [string]$ShardPath = "dist/trained_pure_mlp_shard.bin",
    [string]$ExpectedPath = "dist/trained_pure_mlp_expected.json",
    [string]$EfiPath = "target/x86_64-unknown-uefi/release/neural_box_core.efi"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
$qemuProcess = $null
$client = $null

try {
    if (-not (Test-Path $ShardPath)) {
        throw "Shard not found: $ShardPath"
    }
    if (-not (Test-Path $EfiPath)) {
        throw "UEFI binary not found: $EfiPath"
    }
    if (-not (Test-Path $ExpectedPath)) {
        throw "Expected-vector sidecar not found: $ExpectedPath"
    }

    $shardBytes = [System.IO.File]::ReadAllBytes((Resolve-Path $ShardPath))
    if ($shardBytes.Length -lt 16 -or [System.Text.Encoding]::ASCII.GetString($shardBytes, 0, 4) -ne "NEUR") {
        throw "Shard does not contain a NEUR header"
    }
    $inputDim = [BitConverter]::ToUInt32($shardBytes, 8)
    $quantType = $shardBytes[12]
    $outputDim = [BitConverter]::ToUInt16($shardBytes, 13)
    $hiddenDim = $shardBytes[15]
    if ($inputDim -ne 64 -or $hiddenDim -ne 32 -or $outputDim -ne 16 -or $quantType -ne 0) {
        throw "Shard header is not the supported 64->32->16 ternary MLP"
    }
    if ($shardBytes.Length -lt 656) {
        throw "Dual-layer shard is truncated; expected at least 656 bytes"
    }
    $expected = Get-Content -Raw -Path $ExpectedPath | ConvertFrom-Json
    $testInputs = @($expected.test_batch_i8)
    $testOutputs = @($expected.test_batch_output_i32)
    if ($testInputs.Count -lt 1 -or $testOutputs.Count -ne $testInputs.Count) {
        throw "Expected-vector sidecar dimensions do not match the NEUR shard"
    }
    foreach ($vector in $testInputs) {
        if (@($vector).Count -ne 64) { throw "Each QEMU input vector must have 64 values" }
    }
    foreach ($vector in $testOutputs) {
        if (@($vector).Count -ne $outputDim) { throw "Each expected output must have output_dim values" }
    }
    $shardHash = (Get-FileHash -Algorithm SHA256 -Path $ShardPath).Hash.ToLowerInvariant()
    if ($expected.shard_sha256 -ne $shardHash) {
        throw "Expected-vector sidecar hash does not match the shard being booted"
    }

    New-Item -ItemType Directory -Force -Path "esp/EFI/BOOT", "dist" | Out-Null
    Copy-Item $EfiPath "esp/EFI/BOOT/BOOTX64.EFI" -Force

    $qemu = (Get-Command qemu-system-x86_64 -ErrorAction Stop).Source
    $serialLog = "dist/qemu-uart-com1.log"
    $qemuArgs = @(
        "-bios", "assets/OVMF.fd",
        "-drive", "format=raw,file=fat:rw:esp",
        "-drive", "file=$ShardPath,if=none,id=nvm1,format=raw",
        "-device", "nvme,serial=deadbeef,drive=nvm1",
        "-cpu", $QemuCpu,
        "-net", "none",
        "-display", "none",
        "-monitor", "none",
        "-serial", "file:$serialLog",
        "-serial", "tcp:127.0.0.1:$Port,server=on,wait=off"
    )
    if ($QemuAccel) {
        $qemuArgs = @("-accel", $QemuAccel) + $qemuArgs
    }
    $qemuProcess = Start-Process -FilePath $qemu -ArgumentList $qemuArgs -PassThru -WindowStyle Hidden

    $connectDeadline = [DateTime]::UtcNow.AddSeconds(15)
    while (-not $client -and [DateTime]::UtcNow -lt $connectDeadline) {
        $candidate = [System.Net.Sockets.TcpClient]::new()
        try {
            $connect = $candidate.BeginConnect("127.0.0.1", $Port, $null, $null)
            if ($connect.AsyncWaitHandle.WaitOne(250)) {
                $candidate.EndConnect($connect)
                $client = $candidate
            } else {
                $candidate.Dispose()
            }
        } catch {
            $candidate.Dispose()
        }
        if ($qemuProcess.HasExited) {
            throw "QEMU exited before opening COM2 socket (exit code $($qemuProcess.ExitCode))"
        }
    }
    if (-not $client) {
        throw "Timed out connecting to QEMU COM2 socket on port $Port"
    }
    $stream = $client.GetStream()
    $stream.ReadTimeout = 30000
    $stream.WriteTimeout = 10000

    $readyBytes = [System.Collections.Generic.List[byte]]::new()
    $readyToken = [System.Text.Encoding]::ASCII.GetBytes("UART_READY`n")
    $readyFound = $false
    while ($readyBytes.Count -lt 4096 -and -not $readyFound) {
        $next = $stream.ReadByte()
        if ($next -lt 0) {
            throw "COM2 disconnected before UART_READY"
        }
        $readyBytes.Add([byte]$next)
        if ($readyBytes.Count -ge $readyToken.Length) {
            $match = $true
            $start = $readyBytes.Count - $readyToken.Length
            for ($index = 0; $index -lt $readyToken.Length; $index++) {
                if ($readyBytes[$start + $index] -ne $readyToken[$index]) {
                    $match = $false
                    break
                }
            }
            $readyFound = $match
        }
    }
    if (-not $readyFound) {
        throw "UART_READY handshake was not found in the initial COM2 stream"
    }
    Write-Host "[ROUNDTRIP]: Received COM2 readiness handshake."

    $expectedLength = 3 + (4 * $outputDim)
    for ($frameIndex = 0; $frameIndex -lt $testInputs.Count; $frameIndex++) {
        $inputFrame = New-Object byte[] 66
        $inputFrame[0] = 0x4E
        $inputFrame[1] = 0x42
        for ($index = 0; $index -lt 64; $index++) {
            $value = [int]$testInputs[$frameIndex][$index]
            if ($value -lt -128 -or $value -gt 127) { throw "Test input is outside int8 range" }
            $inputFrame[$index + 2] = [byte]($value -band 0xFF)
        }
        $stream.Write($inputFrame, 0, $inputFrame.Length)
        $stream.Flush()

        $response = New-Object byte[] $expectedLength
        $offset = 0
        while ($offset -lt $response.Length) {
            $received = $stream.Read($response, $offset, $response.Length - $offset)
            if ($received -le 0) {
                throw "COM2 disconnected before output frame $frameIndex was complete"
            }
            $offset += $received
        }

        if ($response[0] -ne 0x4E -or $response[1] -ne 0x52 -or $response[2] -ne $outputDim) {
            throw "Output frame $frameIndex preamble or output_dim did not match"
        }
        $expectedBytes = New-Object byte[] (4 * $outputDim)
        for ($index = 0; $index -lt $outputDim; $index++) {
            $actual = [BitConverter]::ToInt32($response, 3 + ($index * 4))
            $expectedValue = [int]$testOutputs[$frameIndex][$index]
            if ($actual -ne $expectedValue) {
                throw "Frame $frameIndex output $index mismatch: QEMU=$actual, PyTorch=$expectedValue"
            }
            [BitConverter]::GetBytes($expectedValue).CopyTo($expectedBytes, $index * 4)
        }
        for ($index = 0; $index -lt $expectedBytes.Length; $index++) {
            if ($response[$index + 3] -ne $expectedBytes[$index]) {
                throw "Frame $frameIndex output byte $index differs from PyTorch reference bytes"
            }
        }
    }
    Write-Host "[ROUNDTRIP]: QEMU returned $($testInputs.Count) output frames matching PyTorch predictions byte-for-byte."
    Write-Host ("[ROUNDTRIP]: final TX bytes: " + [System.BitConverter]::ToString($response))

    if (-not $qemuProcess.WaitForExit(15000)) {
        throw "QEMU did not exit after the configured one-frame limit"
    }
    if (Test-Path $serialLog) {
        $telemetry = Select-String -Path $serialLog -Pattern "\[GUEST CPUID\]:|\[SIMD\]:|\[SHM\]: Initialized Mailbox|\[MLP VERIFY\]|\[SHARD\]: MAGIC|\[SHARD\]: Layer [12]|\[ALLOC\]:|\[LATENCY\]: shared_mailbox_|\[TIMER\]: Calibrated TSC frequency|\[TIMER\]: Invariant|\[TIMER WARNING\]|\[CPU POWER\]|\[LATENCY\]:|\[UART\]: RX frames=$($testInputs.Count), TX frames=$($testInputs.Count)|\[RING\]: Atomic SPSC"
        $telemetry | ForEach-Object { Write-Host $_.Line }
        if (-not ($telemetry.Line -match "\[SHM\]: Initialized Mailbox at physical addr .*alignment=64")) {
            throw "COM1 log did not confirm an aligned shared mailbox allocation"
        }
        if (-not ($telemetry.Line -match "\[MLP VERIFY\]: Layer1=64->32, hard_sign=32, Layer2=32->16; frames=16, drops=0, scalar_reference=match")) {
            throw "COM1 log did not confirm dual-layer scalar-reference parity and zero drops"
        }
        if (-not ($telemetry.Line -match "\[SHARD\]: MAGIC=0x4E455552, VERSION=1, INPUT_DIM=64, HIDDEN_DIM=32, OUTPUT_DIM=16, QUANT_TYPE=0")) {
            throw "COM1 log did not confirm the dual-layer NEUR header dimensions"
        }
        if (-not ($telemetry.Line -match "\[SHARD\]: Layer 1: 64 -> 32") -or -not ($telemetry.Line -match "\[SHARD\]: Layer 2: 32 -> 16")) {
            throw "COM1 log did not report both dual-layer dimensions"
        }
        if (-not ($telemetry.Line -match "\[TIMER\]: Calibrated TSC frequency")) {
            throw "COM1 log did not contain the TSC calibration report"
        }
        if (-not ($telemetry.Line -match "\[LATENCY\]: turnaround cycles min=")) {
            throw "COM1 log did not contain end-to-end turnaround statistics"
        }
    }
} finally {
    if ($client) {
        $client.Dispose()
    }
    if ($qemuProcess -and -not $qemuProcess.HasExited) {
        Stop-Process -Id $qemuProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}