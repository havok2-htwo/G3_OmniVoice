param(
    [switch]$SkipTorch,
    [switch]$SkipFrontendBuild
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $RepoRoot 'backend'
$FrontendDir = Join-Path $RepoRoot 'frontend'
$FrontendIndex = Join-Path $FrontendDir 'dist\index.html'
$ModelsDir = Join-Path $RepoRoot 'models'
$CondaRoot = 'X:\KI\anaconda3'
$EnvName = 'omnivoice-tts-gui'
$EnvDir = Join-Path $CondaRoot "envs\$EnvName"
$PythonExe = Join-Path $EnvDir 'python.exe'
$CondaExe = Join-Path $CondaRoot 'Scripts\conda.exe'

function Write-Step {
    param([string]$Message)
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-External {
    param(
        [string]$Label,
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory = $RepoRoot
    )

    Write-Step $Label
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Label failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}

function Test-CommandAvailable {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-Path $CondaExe)) {
    throw "Conda was not found at $CondaExe."
}

if (-not (Test-Path $PythonExe)) {
    Invoke-External -Label "Creating Conda environment $EnvName" -FilePath $CondaExe -Arguments @('create', '-y', '-n', $EnvName, 'python=3.12')
}
else {
    Write-Host "Using existing Conda environment: $EnvDir"
}

if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir | Out-Null
}

Invoke-External -Label 'Upgrading pip tooling' -FilePath $PythonExe -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel')

if (-not $SkipTorch) {
    Invoke-External -Label 'Installing CUDA PyTorch' -FilePath $PythonExe -Arguments @(
        '-m', 'pip', 'install', '--upgrade', '--index-url', 'https://download.pytorch.org/whl/cu130',
        'torch==2.10.0+cu130', 'torchvision==0.25.0+cu130', 'torchaudio==2.10.0+cu130'
    )
}
else {
    Write-Host 'Skipping CUDA PyTorch installation.'
}

Invoke-External -Label 'Installing OmniVoice backend package' -FilePath $PythonExe -Arguments @('-m', 'pip', 'install', '-e', "$BackendDir[dev]")

if (-not (Test-CommandAvailable 'npm')) {
    throw 'npm was not found. Install Node.js LTS and re-run install.bat.'
}

Invoke-External -Label 'Installing frontend dependencies' -FilePath 'npm' -Arguments @('install') -WorkingDirectory $FrontendDir

if (-not $SkipFrontendBuild) {
    Invoke-External -Label 'Building frontend' -FilePath 'npm' -Arguments @('run', 'build') -WorkingDirectory $FrontendDir
}
else {
    Write-Host 'Skipping frontend build.'
}

if (-not (Test-Path $FrontendIndex)) {
    throw 'Frontend build output is missing. Expected frontend\dist\index.html.'
}

Write-Host ''
Write-Host 'Installation finished.' -ForegroundColor Green
Write-Host 'Start the app with: .\start_server.bat'
Write-Host 'Open: http://127.0.0.1:8091'
Write-Host ''
Write-Host 'Note: first real OmniVoice startup needs either models\OmniVoice or OMNIVOICE_TTS_ALLOW_MODEL_DOWNLOADS=true.'
