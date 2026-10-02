$bytes = [System.IO.File]::ReadAllBytes('c:\Users\IMOE001\9\scratch\qemu_fat16.img')
$p1StartLba = [BitConverter]::ToUInt32($bytes, 446 + 8)
$p1Sectors = [BitConverter]::ToUInt32($bytes, 446 + 12)
Write-Host "P1 Start LBA: $p1StartLba, Sectors: $p1Sectors"

$p1Offset = $p1StartLba * 512
$secPerClus = $bytes[$p1Offset + 13]
$rsvdSec = [BitConverter]::ToUInt16($bytes, $p1Offset + 14)
$numFats = $bytes[$p1Offset + 16]
$rootEnt = [BitConverter]::ToUInt16($bytes, $p1Offset + 17)
$fatSz16 = [BitConverter]::ToUInt16($bytes, $p1Offset + 22)
Write-Host "FAT parameters: SecPerClus=$secPerClus, RsvdSec=$rsvdSec, NumFats=$numFats, RootEnt=$rootEnt, FatSz16=$fatSz16"

$fatBytes = ($rsvdSec + ($numFats * $fatSz16) + [int]($rootEnt * 32 / 512) + ($secPerClus * 10)) * 512
Write-Host "Active data size in partition: $fatBytes bytes"
