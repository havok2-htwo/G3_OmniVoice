@echo off
setlocal
cd /d "%~dp0"
title G3 OmniVoice Launcher

REM Starts BOTH services in their own named windows:
REM   - Backend  (FastAPI/uvicorn :8091) -> also serves the built UI at /admin and /demo
REM   - Frontend (Vite dev server  :5181) -> hot-reload, only needed for frontend development
REM For normal use the backend alone is enough; the frontend dev server is optional.

echo ============================================================
echo  G3 OmniVoice - starting backend + frontend dev server
echo ------------------------------------------------------------
echo  Backend  : http://127.0.0.1:8091   (API + Admin/Demo UI)
echo  Frontend : http://127.0.0.1:5181   (Vite dev, hot-reload)
echo ============================================================
echo.

start "G3 OmniVoice Backend (API + UI :8091)" cmd /k "%~dp0start_server.bat"
start "G3 OmniVoice Frontend Dev (:5181)" cmd /k "%~dp0start_frontend.bat"

echo Two windows opened. Close them (or press Ctrl+C in each) to stop the services.
echo This launcher window can be closed now.
echo.
pause
