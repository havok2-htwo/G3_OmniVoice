@echo off
setlocal
cd /d "%~dp0frontend"
title G3 OmniVoice Frontend Dev (:5181)

echo Starting G3 OmniVoice Frontend from %CD%...
echo Open: http://127.0.0.1:5181
call .\node_modules\.bin\vite.cmd --host 127.0.0.1 --port 5181 --clearScreen false
pause
