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
$LocalCondaHome = if ($env:OMNIVOICE_TTS_LOCAL_CONDA_HOME) { $env:OMNIVOICE_TTS_LOCAL_CONDA_HOME } else { Join-Path $RepoRoot '.conda' }
$EnvDir = if ($env:OMNIVOICE_TTS_CONDA_ENV_DIR) { $env:OMNIVOICE_TTS_CONDA_ENV_DIR } else { Join-Path $RepoRoot '.conda-env' }
$PythonVersion = if ($env:OMNIVOICE_TTS_CONDA_PYTHON_VERSION) { $env:OMNIVOICE_TTS_CONDA_PYTHON_VERSION } else { '3.12' }
$PythonExe = Join-Path $EnvDir 'python.exe'
$TempDir = Join-Path $RepoRoot '.tmp'

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

function Resolve-Conda {
    if ($env:OMNIVOICE_TTS_CONDA_EXE) {
        return $env:OMNIVOICE_TTS_CONDA_EXE
    }
    if ($env:CONDA_EXE) {
        return $env:CONDA_EXE
    }
    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    $default = 'X:\KI\anaconda3\Scripts\conda.exe'
    if (Test-Path $default) {
        return $default
    }
    throw 'Conda was not found. Install Miniconda/Anaconda, add conda to PATH, or set OMNIVOICE_TTS_CONDA_EXE.'
}

function Use-Local-Conda-State {
    foreach ($path in @(
        $LocalCondaHome,
        (Join-Path $LocalCondaHome 'pkgs'),
        (Join-Path $LocalCondaHome 'envs'),
        (Join-Path $LocalCondaHome 'bld'),
        (Join-Path $LocalCondaHome 'localappdata'),
        (Join-Path $LocalCondaHome 'appdata'),
        $TempDir
    )) {
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path | Out-Null
        }
    }

    $env:LOCALAPPDATA = Join-Path $LocalCondaHome 'localappdata'
    $env:APPDATA = Join-Path $LocalCondaHome 'appdata'
    $env:CONDA_PKGS_DIRS = Join-Path $LocalCondaHome 'pkgs'
    $env:CONDA_ENVS_PATH = Join-Path $LocalCondaHome 'envs'
    $env:CONDA_BLD_PATH = Join-Path $LocalCondaHome 'bld'
    $env:CONDA_NUMBER_CHANNEL_NOTICES = '0'
    $env:CONDA_REPORT_ERRORS = 'false'
    $env:TMP = $TempDir
    $env:TEMP = $TempDir
    $env:TMPDIR = $TempDir
    $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
    $env:PYTHONUTF8 = '1'
}

Use-Local-Conda-State

if ($env:OMNIVOICE_TTS_PYTHON) {
    Invoke-External -Label 'Checking provided OMNIVOICE_TTS_PYTHON' -FilePath $env:OMNIVOICE_TTS_PYTHON -Arguments @(
        '-c',
        'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'
    )
    $PythonExe = $env:OMNIVOICE_TTS_PYTHON
}
else {
    $CondaExe = Resolve-Conda
    if (-not (Test-Path $PythonExe)) {
        Invoke-External -Label "Creating local Conda environment in $EnvDir" -FilePath $CondaExe -Arguments @(
            'create',
            '-y',
            '-p',
            $EnvDir,
            "python=$PythonVersion",
            'pip'
        )
    }
    else {
        Write-Host "Using existing local Conda environment: $EnvDir"
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "Python was not found at $PythonExe."
}

if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir | Out-Null
}

Invoke-External -Label 'Using Python' -FilePath $PythonExe -Arguments @('-c', 'import sys; print(sys.executable); print(sys.version)')

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

Invoke-External -Label 'Installing optional Triton for torch.compile' -FilePath $PythonExe -Arguments @(
    '-m', 'pip', 'install', 'triton-windows<3.7'
)

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
Write-Host "Python: $PythonExe"
Write-Host "Local Conda env: $EnvDir"
Write-Host "Local Conda cache: $LocalCondaHome"
Write-Host 'Start the app with: .\start_server.bat'
Write-Host 'Open: http://127.0.0.1:8091'
Write-Host ''
Write-Host 'Note: first real OmniVoice startup needs either models\OmniVoice or OMNIVOICE_TTS_ALLOW_MODEL_DOWNLOADS=true.'
