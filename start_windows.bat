@echo off
REM Double-click to start the Bird Sound ID website on Windows.
cd /d "%~dp0"
if not exist .venv call setup_windows.bat
.venv\Scripts\python web_app.py
pause
