param(
    [string]$CondaEnv = "SLFNet-rtx30",
    [string]$CudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5",
    [string]$ArchList = "8.6"
)

$ErrorActionPreference = "Stop"

function Find-VsDevCmd {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    $found = Get-ChildItem "${env:ProgramFiles(x86)}\Microsoft Visual Studio" -Recurse -Filter VsDevCmd.bat -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName

    if ($found) {
        return $found
    }

    throw "VsDevCmd.bat not found. Install Visual Studio Build Tools first."
}

if (-not (Test-Path -LiteralPath $CudaHome -PathType Container)) {
    throw "CUDA_HOME not found: $CudaHome"
}

$nvcc = Join-Path $CudaHome "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $nvcc -PathType Leaf)) {
    throw "nvcc.exe not found: $nvcc"
}

$vsDevCmd = Find-VsDevCmd
$deformConvDir = Resolve-Path ".\models\SLFNet\deformconv"

Write-Host "Using Conda env: $CondaEnv"
Write-Host "Using CUDA_HOME: $CudaHome"
Write-Host "Using VS dev cmd: $vsDevCmd"
Write-Host "Using TORCH_CUDA_ARCH_LIST: $ArchList"
Write-Host ""

$cmd = @"
@echo on
call "$vsDevCmd" -arch=amd64 -host_arch=amd64
set CUDA_HOME=$CudaHome
set CUDA_PATH=$CudaHome
set TORCH_CUDA_ARCH_LIST=$ArchList
set DISTUTILS_USE_SDK=1
set MSSdk=1
set PATH=$CudaHome\bin;%PATH%
cd /d "$deformConvDir"
conda run -n $CondaEnv python setup.py build_ext
if errorlevel 1 exit /b %errorlevel%
"@

$bat = Join-Path $env:TEMP "build_dcn_slfnet.cmd"
Set-Content -LiteralPath $bat -Value $cmd -Encoding ASCII
cmd.exe /d /s /c "`"$bat`""
if ($LASTEXITCODE -ne 0) {
    throw "DCN build failed with exit code $LASTEXITCODE"
}

$builtPyd = Get-ChildItem -LiteralPath $deformConvDir -Recurse -Filter "DCN*.pyd" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $builtPyd) {
    throw "DCN build did not produce a .pyd file."
}

$sitePackages = (& conda run -n $CondaEnv python -c "import site; print(site.getsitepackages()[0])" |
    Select-Object -Last 1).Trim()
if (-not (Test-Path -LiteralPath $sitePackages -PathType Container)) {
    throw "Cannot find site-packages for ${CondaEnv}: $sitePackages"
}

Copy-Item -LiteralPath $builtPyd.FullName -Destination (Join-Path $sitePackages $builtPyd.Name) -Force
Write-Host "Installed $($builtPyd.Name) to $sitePackages"

conda run -n $CondaEnv python -c "import DCN; print('DCN import ok')"
if ($LASTEXITCODE -ne 0) {
    throw "DCN import validation failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "DCN build finished successfully."
