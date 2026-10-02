param(
    [int]$Port = 5570,
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

    $qemu = (Get-Command qemu-system-x86_64 -ErrorAction Stop).Source
    $serialLog = "dist/qemu-dual-volume-com1.log"
    $qemuArgs = @(
        "-bios", "assets/OVMF.fd",
        "-drive", "file=$ImagePath,if=none,id=nvm1,format=raw",
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

    $required = @(
        "\[LOADER\]: Read weights.bin from UEFI SimpleFileSystem \(1024 bytes\)",
        "\[LOADER\]: Validated file shard from UEFI FAT volume \(1024 bytes\)",
        "\[LOADER\]: Using model shard from UEFI FAT filesystem",
        "\[SHARD\]: MAGIC=0x4E455552, VERSION=1, INPUT_DIM=64, MODEL_TYPE=0",
        "\[UART\]: RX frames=8, TX frames=8, reset_commands=0, stream_events=8, ring_full_drops=0"
    )
    $telemetry = Select-String -Path $serialLog -Pattern $required
    $telemetry | ForEach-Object { Write-Host $_.Line }
    foreach ($pattern in $required) {
        if (-not (Select-String -Path $serialLog -Pattern $pattern -Quiet)) {
            throw "COM1 log did not contain required filesystem-load/stream pattern: $pattern"
        }
    }
    if (Select-String -Path $serialLog -Pattern "\[LOADER\]: Loaded shard from raw NVMe" -Quiet) {
        throw "Guest loaded from the raw NVMe fallback instead of the FAT filesystem"
    }
    Write-Host "[DUAL VOLUME TEST]: PASS; weights.bin loaded through UEFI SimpleFileSystem, streaming completed with zero drops."
} finally {
    if ($client) { $client.Dispose() }
    if ($qemuProcess -and -not $qemuProcess.HasExited) {
        Stop-Process -Id $qemuProcess.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $qemuProcess.Id -ErrorAction SilentlyContinue
    }
    Pop-Location
}
