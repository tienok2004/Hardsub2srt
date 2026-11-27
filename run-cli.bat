@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Khong tim thay .venv\Scripts\python.exe
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "cli.py" %*
echo.
pause
