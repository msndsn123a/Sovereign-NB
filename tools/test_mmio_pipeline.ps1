param(
    [string]$QemuCpu = "Skylake-Server,+avx512f,+avx512dq",
    [string]$QemuAccel = "",
    [int]$FrameCount = 8,
    [string]$ImagePath = "dist/neural_box_appliance.img",
    [string]$MailboxPath = "dist/ivshmem_mailbox.bin"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
$qemuProcess = $null
try {
    cargo +nightly build --target x86_64-unknown-uefi --release
    if ($LASTEXITCODE -ne 0) { throw "UEFI release build failed: $LASTEXITCODE" }
    python tools/package_image.py
    if ($LASTEXITCODE -ne 0) { throw "Dual-volume image build failed: $LASTEXITCODE" }
    rustc --edition=2021 -O tools/mmio_injector.rs -o dist/mmio_injector.exe
    if ($LASTEXITCODE -ne 0) { throw "MMIO host producer build failed: $LASTEXITCODE" }

    $qemu = (Get-Command qemu-system-x86_64 -ErrorAction Stop).Source
    $objects = & $qemu -object help 2>&1 | Out-String
    $devices = & $qemu -device help 2>&1 | Out-String
    if ($objects -notmatch "memory-backend-file" -or $devices -notmatch "ivshmem-plain") {
        throw "Installed QEMU lacks host-shareable memory-backend-file and/or ivshmem-plain. This build supports only memory-backend-ram; install QEMU with IVSHMEM and file-backed RAM to run the PCI BAR integration test."
    }

    $backing = New-Object byte[] 65536
    [System.IO.File]::WriteAllBytes($MailboxPath, $backing)
    $serialLog = "dist/qemu-mmio-com1.log"
    $qemuArgs = @(
        "-machine", "q35",
        "-m", "512M",
        "-object", "memory-backend-file,id=shmbuf,size=65536,mem-path=$((Resolve-Path $MailboxPath).Path),share=on",
        "-device", "ivshmem-plain,memdev=shmbuf",
        "-bios", "assets/OVMF.fd",
        "-drive", "file=$ImagePath,if=none,id=nvm1,format=raw",
        "-device", "nvme,serial=deadbeef,drive=nvm1",
        "-cpu", $QemuCpu,
        "-net", "none",
        "-display", "none",
        "-monitor", "none",
        "-serial", "file:$serialLog",
        "-serial", "null"
    )
    if ($QemuAccel) { $qemuArgs = @("-accel", $QemuAccel) + $qemuArgs }
    $qemuProcess = Start-Process -FilePath $qemu -ArgumentList $qemuArgs -PassThru -WindowStyle Hidden

    & .\dist\mmio_injector.exe $MailboxPath $FrameCount
    if ($LASTEXITCODE -ne 0) { throw "MMIO producer parity or no-loss check failed: $LASTEXITCODE" }

    if (-not $qemuProcess.WaitForExit(30000)) { throw "QEMU did not stop after the configured MMIO frame count" }
    if ($qemuProcess.ExitCode -ne 0) { throw "QEMU exited with code $($qemuProcess.ExitCode)" }

    $patterns = @(
        "\[PCI MMIO\]: IVSHMEM 1AF4:1110 .*BAR0=",
        "\[MMIO\]: ingress=PCIe shared BAR2",
        "\[MMIO\]: guest_processed=$FrameCount, drops=0",
        "\[MMIO VERIFY\]: guest_processed=$FrameCount, drops=0, max_turnaround_below_1us=true"
    )
    $telemetry = Select-String -Path $serialLog -Pattern $patterns
    $telemetry | ForEach-Object { Write-Host $_.Line }
    foreach ($pattern in $patterns) {
        if (-not (Select-String -Path $serialLog -Pattern $pattern -Quiet)) {
            throw "COM1 log missing required PCI/MMIO verification pattern: $pattern"
        }
    }
    Write-Host "[MMIO TEST]: PASS; direct PCIe BAR round-trip parity, sub-microsecond guest turnaround, and zero packet loss."
} finally {
    if ($qemuProcess -and -not $qemuProcess.HasExited) {
        Stop-Process -Id $qemuProcess.Id -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $qemuProcess.Id -ErrorAction SilentlyContinue
    }
    Pop-Location
}
