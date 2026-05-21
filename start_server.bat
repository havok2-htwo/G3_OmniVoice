@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title G3 OmniVoice Server

set "PYTHON_EXE=X:\KI\anaconda3\envs\omnivoice-tts-gui\python.exe"

if not exist "%PYTHON_EXE%" (
  echo Python Conda environment 'omnivoice-tts-gui' not found.
  echo Run install.bat first.
  echo.
  pause
  exit /b 1
)

set PYTHONUNBUFFERED=1
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
echo Models: %OMNIVOICE_TTS_MODELS_ROOT_DIR%
if exist "%~dp0frontend\dist\index.html" (
  echo Dashboard: http://127.0.0.1:8091
) else (
  echo Frontend build missing. Run install.bat to generate frontend\dist.
  echo API: http://127.0.0.1:8091
)
"%PYTHON_EXE%" -u -m omnivoice_tts_server.main
pause
