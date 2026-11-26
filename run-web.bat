@echo off
rem Double-click to start the Hardsub2srt web UI (no venv activation needed).
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Chua co moi truong .venv. Tao bang:
    echo   py -3 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" app.py %*
if errorlevel 1 pause
