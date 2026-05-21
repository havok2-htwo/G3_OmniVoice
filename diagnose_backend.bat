@echo off
setlocal
set "BASE_URL=http://127.0.0.1:8091"
set "ADMIN_KEY=%OMNIVOICE_TTS_ADMIN_API_KEY%"
if not defined ADMIN_KEY (
  echo Enter admin key for protected checks. Leave empty to skip admin settings.
  set /p ADMIN_KEY=Admin key: 
)

echo Checking G3 OmniVoice backend at %BASE_URL%
curl -s "%BASE_URL%/api/health"
echo.
echo.
if defined ADMIN_KEY (
  echo Checking admin settings
  curl -s -H "X-Admin-Key: %ADMIN_KEY%" "%BASE_URL%/api/admin/settings"
  echo.
) else (
  echo Skipping admin settings because no admin key was provided.
)
pause
