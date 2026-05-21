@echo off
setlocal
set "BASE_URL=http://127.0.0.1:8091"
set "ADMIN_KEY=mein-geheimer-key-1234"

echo Checking G3 OmniVoice backend at %BASE_URL%
curl -s "%BASE_URL%/api/health"
echo.
echo.
echo Checking admin settings
curl -s -H "X-Admin-Key: %ADMIN_KEY%" "%BASE_URL%/api/admin/settings"
echo.
pause
