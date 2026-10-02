param(
    [int]$Port = 5571,
    [ValidateRange(2, 64)][int]$CpuCount = 2,
    [string]$ImagePath = "dist/neural_box_hot_swap.img",
    [string]$UpdateShardPath = "dist/hot_swap_zero_shard.bin",
    [UInt64]$UpdateLba = 264192
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
$qemuProcess = $null
$client = $null

try {
    cargo +nightly build --target x86_64-unknown-uefi --release
    if ($LASTEXITCODE -ne 0) { throw "UEFI release build failed: $LASTEXITCODE" }
    python tools/payload_builder/build_payload.py $UpdateShardPath --model mlp --variant zero
    if ($LASTEXITCODE -ne 0) { throw "Signed zero-model shard generation failed: $LASTEXITCODE" }
    python tools/package_image.py --hot-swap-shard $UpdateShardPath --output $ImagePath
    if ($LASTEXITCODE -ne 0) { throw "Hot-swap image packaging failed: $LASTEXITCODE" }

    $qemu = (Get-Command qemu-system-x86_64 -ErrorAction Stop).Source
    $serialLog = "dist/qemu-hot-swap-com1.log"
    $qemuArgs = @(
        "-bios", "assets/OVMF.fd",
        "-drive", "file=$ImagePath,if=none,id=nvm1,format=raw",
        "-device", "nvme,serial=deadbeef,drive=nvm1",
        "-smp", "$CpuCount",
        "-cpu", "Skylake-Server,+avx512f,+avx512dq",
        "-net", "none",
        "-display", "none",
        "-monitor", "none",
        "-serial", "file:$serialLog",
        "-serial", "tcp:127.0.0.1:$Port,server=on,wait=off"
    )
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
    Write-Host "[HOTSWAP TEST]: UART_READY received."

    function Send-InferenceFrame([int]$frameNumber, [bool]$expectZero) {
        $request = New-Object byte[] 66
        $request[0] = 0x4E
        $request[1] = 0x42
        for ($index = 2; $index -lt $request.Length; $index++) { $request[$index] = 1 }
        $stream.Write($request, 0, $request.Length)
        $stream.Flush()

        $response = New-Object byte[] 67
        $offset = 0
        while ($offset -lt $response.Length) {
            $count = $stream.Read($response, $offset, $response.Length - $offset)
            if ($count -le 0) { throw "COM2 disconnected before response frame $frameNumber" }
            $offset += $count
        }
        if ($response[0] -ne 0x4E -or $response[1] -ne 0x52 -or $response[2] -ne 16) {
            throw "Response frame $frameNumber had an invalid NR header"
        }
        $values = New-Object int[] 16
        $nonzero = 0
        for ($index = 0; $index -lt 16; $index++) {
            $values[$index] = [BitConverter]::ToInt32($response, 3 + ($index * 4))
            if ($values[$index] -ne 0) { $nonzero++ }
        }
        if ($expectZero -and $nonzero -ne 0) {
            throw "Frame $frameNumber used old model output after swap; nonzero lanes=$nonzero"
        }
        if (-not $expectZero -and $nonzero -eq 0) {
            throw "Frame $frameNumber unexpectedly produced all-zero pre-swap output"
        }
        Write-Host "[HOTSWAP TEST]: frame=$frameNumber model=$(if ($expectZero) { 'shadow-zero' } else { 'initial-pattern' }) nonzero_lanes=$nonzero"
    }

    Send-InferenceFrame 1 $false
    Send-InferenceFrame 2 $false

    $update = New-Object byte[] 10
    $update[0] = 0x4E
    $update[1] = 0x55
    [BitConverter]::GetBytes($UpdateLba).CopyTo($update, 2)
    $stream.Write($update, 0, $update.Length)
    $stream.Flush()
    Write-Host "[HOTSWAP TEST]: submitted NU raw-LBA update request at LBA $UpdateLba."

    Send-InferenceFrame 3 $true
    for ($frame = 4; $frame -le 8; $frame++) { Send-InferenceFrame $frame $true }

    if (-not $qemuProcess.WaitForExit(30000)) { throw "QEMU did not exit after 8 streamed inference frames" }
    if ($qemuProcess.ExitCode -ne 0) { throw "QEMU exited with code $($qemuProcess.ExitCode)" }

    $required = @(
        "\[NVMe\]: polled SQ/CQ ready; queue_depth=64, interrupts=disabled, doorbells=MMIO",
        "\[HOTSWAP\]: request queued, lba=$UpdateLba, active_index=0, execution=auxiliary-AP",
        "\[HOTSWAP\]: status=success, lba=$UpdateLba, active_index=1, signature=valid, atomic_order=SeqCst",
        "\[HOTSWAP SUMMARY\]: commands=1, successful=1, rejected=0",
        "\[SMP\]: $($CpuCount - 1)/$($CpuCount - 1) Application Processors awakened and parked",
        "\[TOPOLOGY AUX VERIFY\]: APIC ID=[0-9]+, ring_samples=[1-9][0-9]*, queue_depth=0",
        "\[UART\]: RX frames=8, TX frames=8, reset_commands=0, stream_events=8, ring_full_drops=0"
    )
    foreach ($pattern in $required) {
        if (-not (Select-String -Path $serialLog -Pattern $pattern -Quiet)) {
            throw "COM1 log did not contain hot-swap verification pattern: $pattern"
        }
        (Select-String -Path $serialLog -Pattern $pattern | Select-Object -First 1).Line | Write-Host
    }
    Write-Host "[HOTSWAP TEST]: PASS; inference outputs switched from the active shard to the authenticated shadow shard without dropped frames."
} finally {
    if ($client) { $client.Dispose() }
    if ($qemuProcess -and -not $qemuProcess.HasExited) {
        Stop-Process -Id $qemuProcess.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $qemuProcess.Id -ErrorAction SilentlyContinue
    }
    Pop-Location
}
