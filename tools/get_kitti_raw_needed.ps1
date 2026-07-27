param(
    [Parameter(Mandatory = $true)]
    [string]$DepthRoot,

    [string]$OutDir = ".\kitti_raw_needed"
)

$ErrorActionPreference = "Stop"

$depthRootFull = (Resolve-Path -LiteralPath $DepthRoot).Path
$outDirFull = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutDir)

New-Item -ItemType Directory -Force -Path $outDirFull | Out-Null

$baseUrl = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
$splits = @("train", "val")
$sequences = New-Object System.Collections.Generic.List[string]

foreach ($split in $splits) {
    $splitPath = Join-Path $depthRootFull $split
    if (-not (Test-Path -LiteralPath $splitPath -PathType Container)) {
        Write-Warning "Split folder not found: $splitPath"
        continue
    }

    Get-ChildItem -LiteralPath $splitPath -Directory |
        Where-Object { $_.Name -match "^20\d\d_\d\d_\d\d_drive_\d{4}_sync$" } |
        ForEach-Object { $sequences.Add($_.Name) }
}

$uniqueSequences = $sequences | Sort-Object -Unique
if (-not $uniqueSequences -or $uniqueSequences.Count -eq 0) {
    throw "No KITTI raw sequence folders found under train/val in: $depthRootFull"
}

$rows = foreach ($seqSync in $uniqueSequences) {
    $day = $seqSync.Substring(0, 10)
    $driveNoSync = $seqSync -replace "_sync$", ""
    $zipName = "$seqSync.zip"

    [pscustomobject]@{
        Day = $day
        Sequence = $seqSync
        CalibrationZip = "${day}_calib.zip"
        CalibrationUrl = "$baseUrl/${day}_calib.zip"
        SequenceZip = $zipName
        SequenceUrl = "$baseUrl/$driveNoSync/$zipName"
        ExpectedRawFolder = "$day\$seqSync"
    }
}

$calibUrls = $rows |
    Select-Object CalibrationUrl -Unique |
    Sort-Object CalibrationUrl |
    ForEach-Object { $_.CalibrationUrl }

$sequenceUrls = $rows |
    Sort-Object Sequence |
    ForEach-Object { $_.SequenceUrl }

$allUrls = @($calibUrls + $sequenceUrls)

$rows | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $outDirFull "needed_raw_files.csv")
$allUrls | Set-Content -Encoding UTF8 (Join-Path $outDirFull "needed_raw_urls.txt")

$manual = foreach ($row in ($rows | Sort-Object Sequence)) {
    @"
[$($row.Sequence)]
1. Download calibration once for day: $($row.CalibrationZip)
   $($row.CalibrationUrl)
2. Download sequence zip: $($row.SequenceZip)
   $($row.SequenceUrl)
3. Extract both into your KITTI raw root, so this folder exists:
   <raw_root>\$($row.ExpectedRawFolder)

"@
}

$manual | Set-Content -Encoding UTF8 (Join-Path $outDirFull "manual_download_guide.txt")

Write-Host "Depth root: $depthRootFull"
Write-Host "Sequences found: $($uniqueSequences.Count)"
Write-Host "Output folder: $outDirFull"
Write-Host ""
Write-Host "Created:"
Write-Host "  needed_raw_files.csv"
Write-Host "  needed_raw_urls.txt"
Write-Host "  manual_download_guide.txt"
