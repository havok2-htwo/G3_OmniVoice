@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title G3 OmniVoice Server

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

REM --- Temporary Startup Admin Key ---
if not defined OMNIVOICE_TTS_STARTUP_ADMIN_KEY_TTL_SECONDS set "OMNIVOICE_TTS_STARTUP_ADMIN_KEY_TTL_SECONDS=300"
if not defined OMNIVOICE_TTS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS set "OMNIVOICE_TTS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS=15"
set "OMNIVOICE_TTS_STARTUP_ADMIN_KEY="
set "TMP_DIR=%~dp0.tmp"
set "STARTUP_ADMIN_KEY_FILE=%TMP_DIR%\startup_admin_key.txt"
if not exist "%TMP_DIR%" mkdir "%TMP_DIR%" > nul 2>&1
del /q "%STARTUP_ADMIN_KEY_FILE%" > nul 2>&1
"%PYTHON_EXE%" "%~dp0tools\generate_startup_admin_key.py" > "%STARTUP_ADMIN_KEY_FILE%" 2>nul
if exist "%STARTUP_ADMIN_KEY_FILE%" (
  set /p OMNIVOICE_TTS_STARTUP_ADMIN_KEY=<"%STARTUP_ADMIN_KEY_FILE%"
  del /q "%STARTUP_ADMIN_KEY_FILE%" > nul 2>&1
)
if not defined OMNIVOICE_TTS_ADMIN_API_KEY if defined OMNIVOICE_TTS_STARTUP_ADMIN_KEY (
  set "OMNIVOICE_TTS_ADMIN_API_KEY=%OMNIVOICE_TTS_STARTUP_ADMIN_KEY%"
)

if defined OMNIVOICE_TTS_STARTUP_ADMIN_KEY (
  echo.
  echo ============================================================
  echo Temporary startup admin key ^(valid for %OMNIVOICE_TTS_STARTUP_ADMIN_KEY_TTL_SECONDS% seconds after server start^):
  echo %OMNIVOICE_TTS_STARTUP_ADMIN_KEY%
  echo On a fresh data directory this also becomes the initial persisted admin key.
  echo Copy it now if you need emergency admin access in the browser.
  echo This screen clears automatically in %OMNIVOICE_TTS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS% seconds...
  echo ============================================================
  timeout /t %OMNIVOICE_TTS_STARTUP_ADMIN_KEY_DISPLAY_SECONDS% /nobreak > nul
  cls
) else (
  echo [WARN] Temporary startup admin key could not be generated.
)

echo Starting G3 OmniVoice Server from %CD%...
echo Python: %PYTHON_EXE%
echo Models: %OMNIVOICE_TTS_MODELS_ROOT_DIR%
if exist "%~dp0frontend\dist\index.html" (
  echo Dashboard: http://127.0.0.1:8091
) else (
  echo Frontend build missing. Run install.bat to generate frontend\dist.
  echo API: http://127.0.0.1:8091
)
"%PYTHON_EXE%" -u -m omnivoice_tts_server.main
pause
