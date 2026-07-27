# Run this script from an Administrator PowerShell.
param(
    [string]$CudaVersion = "11.5.0.49613",
    [int]$ChocoTimeoutSeconds = 0,
    [switch]$SkipVS,
    [switch]$SkipCuda
)

$ErrorActionPreference = "Stop"

$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if (-not $isAdmin) {
    throw "Please run PowerShell as Administrator, then run this script again."
}

if (-not (Get-Command choco.exe -ErrorAction SilentlyContinue)) {
    throw "Chocolatey is not available on PATH."
}

if (-not $SkipVS) {
    Write-Host "Installing Visual Studio 2019 Build Tools..."
    choco install visualstudio2019buildtools -y --no-progress --execution-timeout=$ChocoTimeoutSeconds --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --locale en-US"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Visual Studio 2019 Build Tools. Check proxy/network access to https://aka.ms/vs/16/release/channel."
    }

    Write-Host "Installing Visual Studio 2019 C++ workload..."
    choco install visualstudio2019-workload-vctools -y --no-progress --execution-timeout=$ChocoTimeoutSeconds
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Visual Studio 2019 C++ workload."
    }
}

if (-not $SkipCuda) {
    Write-Host "Installing CUDA Toolkit $CudaVersion..."
    $cudaInstaller = Join-Path $env:TEMP "chocolatey\cuda\$CudaVersion\cuda_11.5.0_496.13_win10.exe"
    if ((Test-Path -LiteralPath $cudaInstaller -PathType Leaf) -and ($CudaVersion -eq "11.5.0.49613")) {
        Write-Host "Found cached CUDA installer: $cudaInstaller"
        Write-Host "Running local CUDA installer. This can take several minutes..."
        $cudaProcess = Start-Process -FilePath $cudaInstaller -ArgumentList "/s" -Wait -PassThru -WindowStyle Hidden
        if ($cudaProcess.ExitCode -ne 0) {
            throw "CUDA local installer failed with exit code $($cudaProcess.ExitCode)."
        }
    } else {
        choco install cuda --version=$CudaVersion -y --no-progress --execution-timeout=$ChocoTimeoutSeconds
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install CUDA Toolkit $CudaVersion."
        }
    }

    $nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin\nvcc.exe"
    if (-not (Test-Path -LiteralPath $nvcc -PathType Leaf)) {
        throw "CUDA install did not create nvcc.exe at $nvcc."
    }
}

Write-Host ""
Write-Host "Done. Restart PowerShell before compiling DCN."
