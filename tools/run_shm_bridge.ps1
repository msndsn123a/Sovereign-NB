param(
    [int]$Port = 5568,
    [string]$ShardPath = "dist/trained_pure_mlp_shard.bin",
    [string]$ExpectedPath = "dist/trained_pure_mlp_expected.json",
    [string]$MailboxPath = "dist/shm_mailbox.bin",
    [string]$QemuAccel = "whpx"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repoRoot
$qemuProcess = $null
try {
    foreach ($path in @($ShardPath, $ExpectedPath)) {
        if (-not (Test-Path $path)) { throw "Required trained artifact missing: $path" }
    }

    cargo +nightly build --features host-ipc --target x86_64-unknown-uefi --release
    if ($LASTEXITCODE -ne 0) { throw "host-ipc UEFI build failed: $LASTEXITCODE" }
    rustc --edition=2021 -O tools/host_injector.rs -o dist/host_injector.exe
    if ($LASTEXITCODE -ne 0) { throw "Windows host injector build failed: $LASTEXITCODE" }

    $zeroWindow = New-Object byte[] 65536
    [System.IO.File]::WriteAllBytes($MailboxPath, $zeroWindow)
    New-Item -ItemType Directory -Force -Path "esp/EFI/BOOT", "dist" | Out-Null
    Copy-Item target/x86_64-unknown-uefi/release/neural_box_core.efi esp/EFI/BOOT/BOOTX64.EFI -Force

    $qemu = (Get-Command qemu-system-x86_64 -ErrorAction Stop).Source
    $qemuObjects = & $qemu -object help 2>&1 | Out-String
    if ($qemuObjects -notmatch "memory-backend-file") {
        throw "Installed QEMU does not register memory-backend-file (its object list only includes memory-backend-ram); a Windows file-backed guest RAM bridge cannot be created with this binary. Install/build QEMU with host file-backed RAM support or ivshmem and rerun."
    }

    $serialLog = "dist/qemu-hostipc-com1.log"
    $qemuArgs = @(
        "-machine", "q35",
        "-m", "512M,maxmem=5G,slots=1",
        "-object", "memory-backend-file,id=shmbuf,size=65536,mem-path=$((Resolve-Path $MailboxPath).Path),share=on",
        "-device", "pc-dimm,memdev=shmbuf,addr=0x100000000",
        "-bios", "assets/OVMF.fd",
        "-drive", "format=raw,file=fat:rw:esp",
        "-drive", "file=$ShardPath,if=none,id=nvm1,format=raw",
        "-device", "nvme,serial=deadbeef,drive=nvm1",
        "-cpu", "Skylake-Server,+avx512f,+avx512dq",
        "-net", "none",
        "-display", "none",
        "-monitor", "none",
        "-serial", "file:$serialLog",
        "-serial", "null"
    )
    if ($QemuAccel) { $qemuArgs = @("-accel", $QemuAccel) + $qemuArgs }
    $qemuProcess = Start-Process -FilePath $qemu -ArgumentList $qemuArgs -PassThru -WindowStyle Hidden

    & .\dist\host_injector.exe $MailboxPath $ExpectedPath
    if ($LASTEXITCODE -ne 0) { throw "Host IPC injector failed: $LASTEXITCODE" }

    if (-not $qemuProcess.WaitForExit(30000)) {
        throw "QEMU guest did not exit after the configured host IPC frame count"
    }
    if (Test-Path $serialLog) {
        $patterns = @("\[SHM\]: Initialized Mailbox", "\[SHM HOST IPC\]", "\[LATENCY\]: host_ipc_", "\[SHARD\]: MAGIC", "\[SHARD\]: Layer [12]", "\[SIMD\]:")
        $telemetry = Select-String -Path $serialLog -Pattern $patterns
        $telemetry | ForEach-Object { Write-Host $_.Line }
        if (-not ($telemetry.Line -match "\[SHM HOST IPC\]: guest_processed=8, drops=0")) {
            throw "Guest COM1 log does not confirm 8 host frames and zero drops"
        }
    } else {
        throw "COM1 guest log not created: $serialLog"
    }
} finally {
    if ($qemuProcess -and -not $qemuProcess.HasExited) {
        Stop-Process -Id $qemuProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Pop-Location
}
