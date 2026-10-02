param(
    [int]$Port = 5570,
    [ValidateRange(1, 64)][int]$CpuCount = 1,
    [switch]$TamperWeights,
    [switch]$RequireSparseActivation,
    [string]$QemuAccel = "",
    [string]$QemuCpu = "Skylake-Server,+avx512f,+avx512dq",
    [string]$ImagePath = "dist/neural_box_appliance.img"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
$qemuProcess = $null
$client = $null

try {
    if (-not (Test-Path $ImagePath)) { throw "Dual-volume image not found: $ImagePath" }
    $efiPath = "target/x86_64-unknown-uefi/release/neural_box_core.efi"
    if (-not (Test-Path $efiPath)) { throw "UEFI binary not found: $efiPath" }

    if ($TamperWeights) {
        $validShard = "dist/production_shard.bin"
        if (-not (Test-Path $validShard)) { throw "Signed test shard not found: $validShard" }
        $mutated = [System.IO.File]::ReadAllBytes((Resolve-Path $validShard))
        if ($mutated.Length -lt 80 -or [BitConverter]::ToUInt32($mutated, 4) -ne 2) {
            throw "Tamper test requires a signed NEUR v2 shard"
        }
        $mutated[80] = [byte]($mutated[80] -bxor 0x01)
        $tamperedPath = "dist/tampered_weights.bin"
        [System.IO.File]::WriteAllBytes($tamperedPath, $mutated)
        $ImagePath = "dist/neural_box_tampered.img"
        python tools/package_image.py --shard-path $tamperedPath --output $ImagePath
        if ($LASTEXITCODE -ne 0) { throw "Tampered image packaging failed: $LASTEXITCODE" }
        Write-Host "[SECURITY TEST]: Mutated payload byte; boot must reject the shard signature."
    }
    if ($TamperWeights -and $RequireSparseActivation) {
        throw "Sparse activation validation cannot be combined with the tamper test"
    }

    $qemu = (Get-Command qemu-system-x86_64 -ErrorAction Stop).Source
    $serialLog = if ($RequireSparseActivation) {
        "dist/qemu-dual-volume-sparse-com1.log"
    } else {
        "dist/qemu-dual-volume-com1.log"
    }
    $qemuArgs = @(
        "-bios", "assets/OVMF.fd",
        "-drive", "file=$ImagePath,if=none,id=nvm1,format=raw",
        "-device", "nvme,serial=deadbeef,drive=nvm1",
        "-smp", "$CpuCount",
        "-cpu", $QemuCpu,
        "-net", "none",
        "-display", "none",
        "-monitor", "none",
        "-serial", "file:$serialLog",
        "-serial", "tcp:127.0.0.1:$Port,server=on,wait=off"
    )
    if ($QemuAccel) { $qemuArgs = @("-accel", $QemuAccel) + $qemuArgs }
    $qemuProcess = Start-Process -FilePath $qemu -ArgumentList $qemuArgs -PassThru -WindowStyle Hidden

    $connectDeadline = [DateTime]::UtcNow.AddSeconds(25)
    while (-not $client -and [DateTime]::UtcNow -lt $connectDeadline) {
        $candidate = [System.Net.Sockets.TcpClient]::new()
        try {
            $connect = $candidate.BeginConnect("127.0.0.1", $Port, $null, $null)
            if ($connect.AsyncWaitHandle.WaitOne(250)) {
                $candidate.EndConnect($connect)
                $client = $candidate
            } else { $candidate.Dispose() }
        } catch { $candidate.Dispose() }
        if ($qemuProcess.HasExited) { throw "QEMU exited before COM2 became available (exit $($qemuProcess.ExitCode))" }
    }
    if (-not $client) { throw "Timed out waiting for QEMU COM2 on port $Port" }

    $stream = $client.GetStream()
    $stream.ReadTimeout = 30000
    $stream.WriteTimeout = 10000
    $ready = [System.Collections.Generic.List[byte]]::new()
    $readyToken = [System.Text.Encoding]::ASCII.GetBytes("UART_READY`n")
    $readyFound = $false
    while ($ready.Count -lt 4096 -and -not $readyFound) {
        $value = $stream.ReadByte()
        if ($value -lt 0) { throw "COM2 disconnected before UART_READY" }
        $ready.Add([byte]$value)
        if ($ready.Count -ge $readyToken.Length) {
            $offset = $ready.Count - $readyToken.Length
            $readyFound = $true
            for ($index = 0; $index -lt $readyToken.Length; $index++) {
                if ($ready[$offset + $index] -ne $readyToken[$index]) { $readyFound = $false; break }
            }
        }
    }
    if (-not $readyFound) { throw "UART_READY handshake was not received" }
    Write-Host "[DUAL VOLUME TEST]: UART_READY received after GPT image boot."

    for ($frame = 1; $frame -le 8; $frame++) {
        $request = New-Object byte[] 66
        $request[0] = 0x4E
        $request[1] = 0x42
        $stream.Write($request, 0, $request.Length)
        $stream.Flush()

        $response = New-Object byte[] 67
        $offset = 0
        while ($offset -lt $response.Length) {
            $count = $stream.Read($response, $offset, $response.Length - $offset)
            if ($count -le 0) { throw "COM2 disconnected before response frame $frame" }
            $offset += $count
        }
        if ($response[0] -ne 0x4E -or $response[1] -ne 0x52 -or $response[2] -ne 16) {
            throw "Response frame $frame did not contain NR + output_dim=16"
        }
        for ($index = 0; $index -lt 16; $index++) {
            if ([BitConverter]::ToInt32($response, 3 + ($index * 4)) -ne 0) {
                throw "Default identity-safe MLP expected zero output for zero input (frame $frame, output $index)"
            }
        }
    }
    Write-Host "[DUAL VOLUME TEST]: Received 8 valid inference responses."

    if (-not $qemuProcess.WaitForExit(30000)) { throw "QEMU did not exit after 8 stream frames" }
    if ($qemuProcess.ExitCode -ne 0) { throw "QEMU exited with code $($qemuProcess.ExitCode)" }

    $expectedAps = $CpuCount - 1
    $required = @(
        "\[LOADER\]: Read weights.bin from UEFI SimpleFileSystem \(1024 bytes\)",
        "\[SECURITY (TME|SME|MEM)\]:",
        "\[SECURITY GATE\]: strict_memory_encryption=false, policy_satisfied=(true|false)",
        "\[PAGING\]: Custom CR3 activated .*encrypted huge pages=[0-9]+, C-bit mask=0x[0-9a-fA-F]{16}",
        "\[SMP\]: $expectedAps/$expectedAps Application Processors awakened and parked",
        "\[WATCHDOG\]: WDAT unavailable; safe no-op fallback \(no chipset TCO base guessed\)\.",
        "\[TOPOLOGY\]: Uniform/Symmetric",
        "\[UART\]: RX frames=8, TX frames=8, reset_commands=0, stream_events=8, ring_full_drops=0"
    )
    if ($CpuCount -gt 1) {
        $required += "\[TOPOLOGY AUX VERIFY\]: APIC ID=[0-9]+, ring_samples=[1-9][0-9]*, queue_depth=[0-9]+"
    }
    if ($TamperWeights) {
        $required += @(
            "\[SECURITY\]: Signature mismatch or invalid header - rejecting shard",
            "\[SECURITY\]: Rejected unsigned or tampered shard; using safe identity model\."
        )
    } else {
        $required += @(
            "\[LOADER\]: Validated file shard from UEFI FAT volume \(1024 bytes\)",
            "\[LOADER\]: Using model shard from UEFI FAT filesystem",
            "\[SECURITY\]: Shard signature valid \(Ed25519 verified\)",
            "\[SHARD\]: MAGIC=0x4E455552, VERSION=2, INPUT_DIM=64, MODEL_TYPE=0"
        )
        if ($RequireSparseActivation) {
            $required += @(
                "\[SHARD CONFIG\]: multi_stream=false, quant_type=0, pot_scale=0, block_sparse=true, activation_lut=true",
                "\[SPARSE VERIFY\]: dense_scalar=match, sparse_scalar=match, vector=(match|disabled), active_blocks=160/320",
                "\[MLP KERNEL\]: evaluator=block-sparse-(scalar|avx512), active_blocks=160/320, skipped_blocks=160, activation_lut=true",
                "\[ACTIVATION LUT\]: integer SiLU Q4, entries=256, bytes=512, align=64, initial_L1_warm_lines=8, checksum=-?[0-9]+, enabled=true"
            )
        }
    }
    $telemetry = Select-String -Path $serialLog -Pattern $required
    $telemetry | ForEach-Object { Write-Host $_.Line }
    foreach ($pattern in $required) {
        if (-not (Select-String -Path $serialLog -Pattern $pattern -Quiet)) {
            throw "COM1 log did not contain required filesystem-load/stream pattern: $pattern"
        }
    }
    $cycleTelemetry = Select-String -Path $serialLog -Pattern "\[TELEMETRY\]: frames=8, min_cycles=[0-9]+, max_cycles=[0-9]+, avg_cycles=[0-9]+"
    if (-not $cycleTelemetry) { throw "COM1 log did not include the eight-frame cycle benchmark" }
    Write-Host $cycleTelemetry.Line
    $powerTelemetry = Select-String -Path $serialLog -Pattern "\[POWER\]: WAITPKG (active \(UMONITOR/UMWAIT\)|unsupported, falling back to PAUSE loop)"
    if (-not $powerTelemetry) { throw "COM1 log did not report WAITPKG activation or fallback" }
    Write-Host $powerTelemetry.Line
    if (-not $TamperWeights -and (Select-String -Path $serialLog -Pattern "\[LOADER\]: Loaded shard from raw NVMe" -Quiet)) {
        throw "Guest loaded from the raw NVMe fallback instead of the FAT filesystem"
    }
    if ($TamperWeights) {
        if (Select-String -Path $serialLog -Pattern "\[SECURITY\]: Shard signature valid" -Quiet) {
            throw "Tampered model was incorrectly accepted"
        }
        Write-Host "[SECURITY TEST]: PASS; mutated shard rejected and safe identity fallback completed with zero drops."
    } else {
        if ($RequireSparseActivation) {
            Write-Host "[DUAL VOLUME TEST]: PASS; sparse blocks skipped, LUT activation warmed and evaluated, dense/sparse parity verified, streaming completed with zero drops."
        } else {
            Write-Host "[DUAL VOLUME TEST]: PASS; signed weights.bin authenticated through UEFI SimpleFileSystem and streaming completed with zero drops."
        }
    }
} finally {
    if ($client) { $client.Dispose() }
    if ($qemuProcess -and -not $qemuProcess.HasExited) {
        Stop-Process -Id $qemuProcess.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $qemuProcess.Id -ErrorAction SilentlyContinue
    }
    Pop-Location
}
