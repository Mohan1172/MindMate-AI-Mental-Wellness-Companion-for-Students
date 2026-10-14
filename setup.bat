@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo        MindMate First-Time Setup
echo ========================================
echo.

echo Creating the project virtual environment and installing dependencies...
python -m venv .venv
if errorlevel 1 (
    echo.
    echo Could not create .venv. Make sure Python is installed with the standard venv component.
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Setup completed successfully.
echo Now run app.py in VS Code or double-click run.bat.
pause
exit /b 0

:failed
echo.
echo Setup failed. Check your internet connection and try again.
pause
exit /b 1
