@echo off
REM One-time setup for Windows: creates .venv and installs everything.
cd /d "%~dp0"
py -m venv .venv || python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
echo.
echo Setup finished. Double-click "start_windows.bat" to open the website.
pause
