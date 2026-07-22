@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title G3 OmniVoice Backend (API + UI :8091)

set "ROOT=%~dp0"
set "LOCAL_CONDA_HOME=%ROOT%.conda"
set "CONDA_ENV_DIR=%ROOT%.conda-env"
set "PYTHON_EXE="

if defined OMNIVOICE_TTS_LOCAL_CONDA_HOME set "LOCAL_CONDA_HOME=%OMNIVOICE_TTS_LOCAL_CONDA_HOME%"
if defined OMNIVOICE_TTS_CONDA_ENV_DIR set "CONDA_ENV_DIR=%OMNIVOICE_TTS_CONDA_ENV_DIR%"
if defined OMNIVOICE_TTS_PYTHON set "PYTHON_EXE=%OMNIVOICE_TTS_PYTHON%"
if not defined PYTHON_EXE if exist "%CONDA_ENV_DIR%\python.exe" set "PYTHON_EXE=%CONDA_ENV_DIR%\python.exe"

if not defined PYTHON_EXE (
  echo Local Python environment not found at "%CONDA_ENV_DIR%".
  echo Run install.bat first.
  echo.
  pause
  exit /b 1
)
if not exist "%PYTHON_EXE%" (
  echo Python executable not found: "%PYTHON_EXE%"
  echo Run install.bat first or set OMNIVOICE_TTS_PYTHON.
  echo.
  pause
  exit /b 1
)

set PYTHONUNBUFFERED=1
set PYTHONUTF8=1
set PIP_DISABLE_PIP_VERSION_CHECK=1
if not exist "%LOCAL_CONDA_HOME%\pkgs" mkdir "%LOCAL_CONDA_HOME%\pkgs" > nul 2>&1
if not exist "%LOCAL_CONDA_HOME%\envs" mkdir "%LOCAL_CONDA_HOME%\envs" > nul 2>&1
if not exist "%LOCAL_CONDA_HOME%\bld" mkdir "%LOCAL_CONDA_HOME%\bld" > nul 2>&1
if not exist "%LOCAL_CONDA_HOME%\localappdata" mkdir "%LOCAL_CONDA_HOME%\localappdata" > nul 2>&1
if not exist "%LOCAL_CONDA_HOME%\appdata" mkdir "%LOCAL_CONDA_HOME%\appdata" > nul 2>&1
set "LOCALAPPDATA=%LOCAL_CONDA_HOME%\localappdata"
set "APPDATA=%LOCAL_CONDA_HOME%\appdata"
set "CONDA_PKGS_DIRS=%LOCAL_CONDA_HOME%\pkgs"
set "CONDA_ENVS_PATH=%LOCAL_CONDA_HOME%\envs"
set "CONDA_BLD_PATH=%LOCAL_CONDA_HOME%\bld"
set "CONDA_NUMBER_CHANNEL_NOTICES=0"
set "CONDA_REPORT_ERRORS=false"
if not defined OMNIVOICE_TTS_RUNTIME_BACKEND set "OMNIVOICE_TTS_RUNTIME_BACKEND=omnivoice"
if not defined OMNIVOICE_TTS_MODELS_ROOT_DIR set "OMNIVOICE_TTS_MODELS_ROOT_DIR=%~dp0models"
if not defined OMNIVOICE_TTS_ALLOW_MODEL_DOWNLOADS set "OMNIVOICE_TTS_ALLOW_MODEL_DOWNLOADS=false"
if not defined OMNIVOICE_TTS_PORT set "OMNIVOICE_TTS_PORT=8091"

echo Starting G3 OmniVoice Server from %CD%...
echo Python: %PYTHON_EXE%
echo Models: %OMNIVOICE_TTS_MODELS_ROOT_DIR%
if exist "%~dp0frontend\dist\index.html" (
  echo Dashboard: http://127.0.0.1:8091
  echo Admin login: username/password ^(default first run: admin/admin^)
) else (
  echo Frontend build missing. Run install.bat to generate frontend\dist.
  echo API: http://127.0.0.1:8091
)
REM Shrink the CUDA caching-allocator reserved pool (less fragmentation -> lower idle VRAM,
REM leaves room for Whisper + Ollama). expandable_segments is Linux-only / ignored on Windows.
set "PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256"
"%PYTHON_EXE%" -u -m omnivoice_tts_server.main
pause
