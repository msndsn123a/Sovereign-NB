param(
    [int]$Port = 5569,
    [string]$QemuAccel = "",
    [string]$QemuCpu = "Skylake-Server,+avx512f,+avx512dq",
    [string]$ShardPath = "dist/attention_test_shard.bin",
    [string]$EfiPath = "target/x86_64-unknown-uefi/release/neural_box_core.efi"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
$qemuProcess = $null
$client = $null

try {
    cargo +nightly build --target x86_64-unknown-uefi --release
    if ($LASTEXITCODE -ne 0) { throw "UEFI release build failed: $LASTEXITCODE" }
    & "$PSScriptRoot/payload_builder/build_payload.ps1" -outPath $ShardPath -model attention
    if ($LASTEXITCODE -ne 0) { throw "Attention shard generation failed: $LASTEXITCODE" }
    if (-not (Test-Path $EfiPath)) { throw "UEFI binary not found: $EfiPath" }

    New-Item -ItemType Directory -Force -Path "esp/EFI/BOOT", "dist" | Out-Null
    Copy-Item $EfiPath "esp/EFI/BOOT/BOOTX64.EFI" -Force
    $qemu = (Get-Command qemu-system-x86_64 -ErrorAction Stop).Source
    $serialLog = "dist/qemu-attention-sequence-com1.log"
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
    if ($QemuAccel) { $qemuArgs = @("-accel", $QemuAccel) + $qemuArgs }
    $qemuProcess = Start-Process -FilePath $qemu -ArgumentList $qemuArgs -PassThru -WindowStyle Hidden

    $connectDeadline = [DateTime]::UtcNow.AddSeconds(20)
    while (-not $client -and [DateTime]::UtcNow -lt $connectDeadline) {
        $candidate = [System.Net.Sockets.TcpClient]::new()
        try {
            $connect = $candidate.BeginConnect("127.0.0.1", $Port, $null, $null)
            if ($connect.AsyncWaitHandle.WaitOne(250)) {
                $candidate.EndConnect($connect)
                $client = $candidate
            } else { $candidate.Dispose() }
        } catch { $candidate.Dispose() }
        if ($qemuProcess.HasExited) { throw "QEMU exited before opening COM2 (exit $($qemuProcess.ExitCode))" }
    }
    if (-not $client) { throw "Timed out connecting to QEMU COM2 on port $Port" }

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
    if (-not $readyFound) { throw "UART_READY handshake was not found" }
    Write-Host "[ATTENTION TEST]: UART_READY received."

    for ($frame = 1; $frame -le 3; $frame++) {
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
            if ($count -le 0) { throw "COM2 disconnected before attention output frame $frame" }
            $offset += $count
        }
        if ($response[0] -ne 0x4E -or $response[1] -ne 0x52 -or $response[2] -ne 16) {
            throw "Attention frame $frame response header mismatch"
        }
        $expected = 4194304 * $frame
        for ($index = 0; $index -lt 16; $index++) {
            $actual = [BitConverter]::ToInt32($response, 3 + ($index * 4))
            if ($actual -ne $expected) { throw "Frame $frame output $index mismatch: got $actual, expected $expected" }
        }
        Write-Host "[ATTENTION TEST]: frame=$frame state00=$($frame * 4096) output0=$expected"
    }

    # Standalone NR is a control event (no length/payload and no response frame).
    $reset = [byte[]](0x4E, 0x52)
    $stream.Write($reset, 0, $reset.Length)
    $stream.Flush()
    Write-Host "[ATTENTION TEST]: sent NR reset command after three input frames."

    if (-not $qemuProcess.WaitForExit(15000)) { throw "QEMU did not exit after the NR reset event" }
    if ($qemuProcess.ExitCode -ne 0) { throw "QEMU exited with code $($qemuProcess.ExitCode)" }

    $required = @(
        "\[SHARD\]: .*MODEL_TYPE=1.*OUTPUT_DIM=16",
        "\[SHARD\]: Model=causal-linear-attention",
        "\[ATTENTION STATE\]: frame=1 state00=4096",
        "\[ATTENTION STATE\]: frame=2 state00=8192",
        "\[ATTENTION STATE\]: frame=3 state00=12288",
        "\[ATTENTION RESET\]: command=NR state00=0 reset_count=1",
        "\[UART\]: RX frames=3, TX frames=3, reset_commands=1, stream_events=4, ring_full_drops=0"
    )
    $matched = Select-String -Path $serialLog -Pattern $required
    $matched | ForEach-Object { Write-Host $_.Line }
    foreach ($pattern in $required) {
        if (-not (Select-String -Path $serialLog -Pattern $pattern -Quiet)) {
            throw "COM1 log did not contain required verification pattern: $pattern"
        }
    }
    Write-Host "[ATTENTION TEST]: PASS; recurrent state accumulated across 3 frames and reset to zero."
} finally {
    if ($client) { $client.Dispose() }
    if ($qemuProcess -and -not $qemuProcess.HasExited) {
        Stop-Process -Id $qemuProcess.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $qemuProcess.Id -ErrorAction SilentlyContinue
    }
    Pop-Location
}
