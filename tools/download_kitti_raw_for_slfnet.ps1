param(
    [Parameter(Mandatory = $true)]
    [string]$DepthRoot,

    [Parameter(Mandatory = $true)]
    [string]$RawRoot,

    [string]$CacheDir = "",

    [int]$Retries = 3,

    [switch]$Force,

    [switch]$RemoveZipAfterExtract
)

$ErrorActionPreference = "Stop"

function Resolve-OrCreateDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }

    return (Resolve-Path -LiteralPath $Path).Path
}

function Get-KittiDepthSequences {
    param([Parameter(Mandatory = $true)][string]$DepthRootFull)

    $splits = @("train", "val")
    $sequences = New-Object System.Collections.Generic.List[string]

    foreach ($split in $splits) {
        $splitPath = Join-Path $DepthRootFull $split
        if (-not (Test-Path -LiteralPath $splitPath -PathType Container)) {
            Write-Warning "Split folder not found: $splitPath"
            continue
        }

        Get-ChildItem -LiteralPath $splitPath -Directory |
            Where-Object { $_.Name -match "^20\d\d_\d\d_\d\d_drive_\d{4}_sync$" } |
            ForEach-Object { $sequences.Add($_.Name) }
    }

    $unique = @($sequences | Sort-Object -Unique)
    if ($unique.Count -eq 0) {
        throw "No KITTI raw sequence folders found under train/val in: $DepthRootFull"
    }

    return $unique
}

function Invoke-DownloadWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [Parameter(Mandatory = $true)][int]$Retries
    )

    $parent = Split-Path -Parent $OutFile
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            Write-Host "Downloading ($attempt/$Retries): $Url"

            if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
                & curl.exe -L --fail --retry 3 --retry-delay 5 --continue-at - --output "$OutFile" "$Url"
                if ($LASTEXITCODE -ne 0) {
                    throw "curl.exe failed with exit code $LASTEXITCODE"
                }
            }
            else {
                Invoke-WebRequest -Uri $Url -OutFile $OutFile
            }

            if (-not (Test-Path -LiteralPath $OutFile -PathType Leaf)) {
                throw "Download did not create file: $OutFile"
            }

            if ((Get-Item -LiteralPath $OutFile).Length -le 0) {
                throw "Downloaded file is empty: $OutFile"
            }

            return
        }
        catch {
            Write-Warning $_.Exception.Message
            if ($attempt -eq $Retries) {
                throw
            }
            Start-Sleep -Seconds (5 * $attempt)
        }
    }
}

function Expand-ZipToRawRoot {
    param(
        [Parameter(Mandatory = $true)][string]$ZipPath,
        [Parameter(Mandatory = $true)][string]$RawRootFull
    )

    Write-Host "Extracting: $ZipPath"
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $RawRootFull -Force
}

$depthRootFull = (Resolve-Path -LiteralPath $DepthRoot).Path
$rawRootFull = Resolve-OrCreateDirectory -Path $RawRoot

if ([string]::IsNullOrWhiteSpace($CacheDir)) {
    $CacheDir = Join-Path $rawRootFull "_download_cache"
}
$cacheDirFull = Resolve-OrCreateDirectory -Path $CacheDir

$baseUrl = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
$sequences = Get-KittiDepthSequences -DepthRootFull $depthRootFull
$days = @($sequences | ForEach-Object { $_.Substring(0, 10) } | Sort-Object -Unique)

Write-Host "Depth root: $depthRootFull"
Write-Host "Raw root:   $rawRootFull"
Write-Host "Cache dir:  $cacheDirFull"
Write-Host "Sequences:  $($sequences.Count)"
Write-Host "Days:       $($days -join ', ')"
Write-Host ""

foreach ($day in $days) {
    $expectedCalib = Join-Path $rawRootFull "$day\calib_cam_to_cam.txt"
    if ((Test-Path -LiteralPath $expectedCalib -PathType Leaf) -and -not $Force) {
        Write-Host "Calibration exists, skipping: $expectedCalib"
        continue
    }

    $calibZip = "${day}_calib.zip"
    $calibUrl = "$baseUrl/$calibZip"
    $calibZipPath = Join-Path $cacheDirFull $calibZip

    if (-not (Test-Path -LiteralPath $calibZipPath -PathType Leaf) -or $Force) {
        Invoke-DownloadWithRetry -Url $calibUrl -OutFile $calibZipPath -Retries $Retries
    }
    else {
        Write-Host "Using cached zip: $calibZipPath"
    }

    Expand-ZipToRawRoot -ZipPath $calibZipPath -RawRootFull $rawRootFull

    if ($RemoveZipAfterExtract) {
        Remove-Item -LiteralPath $calibZipPath -Force
    }
}

foreach ($seqSync in $sequences) {
    $day = $seqSync.Substring(0, 10)
    $expectedSequence = Join-Path $rawRootFull "$day\$seqSync"
    if ((Test-Path -LiteralPath $expectedSequence -PathType Container) -and -not $Force) {
        Write-Host "Sequence exists, skipping: $expectedSequence"
        continue
    }

    $driveNoSync = $seqSync -replace "_sync$", ""
    $seqZip = "$seqSync.zip"
    $seqUrl = "$baseUrl/$driveNoSync/$seqZip"
    $seqZipPath = Join-Path $cacheDirFull $seqZip

    if (-not (Test-Path -LiteralPath $seqZipPath -PathType Leaf) -or $Force) {
        Invoke-DownloadWithRetry -Url $seqUrl -OutFile $seqZipPath -Retries $Retries
    }
    else {
        Write-Host "Using cached zip: $seqZipPath"
    }

    Expand-ZipToRawRoot -ZipPath $seqZipPath -RawRootFull $rawRootFull

    if (-not (Test-Path -LiteralPath $expectedSequence -PathType Container)) {
        throw "Expected sequence folder was not created: $expectedSequence"
    }

    if ($RemoveZipAfterExtract) {
        Remove-Item -LiteralPath $seqZipPath -Force
    }
}

Write-Host ""
Write-Host "Done. KITTI Raw data is ready at:"
Write-Host "  $rawRootFull"
Write-Host ""
Write-Host "Next step:"
Write-Host "  python data_prepare\prepare_KITTI_DC.py --path_root_dc `"$depthRootFull`" --path_root_raw `"$rawRootFull`""
