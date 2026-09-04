@echo off
REM Run as administrator. Windows prompts for credentials; none go into this file.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_autopilot_tasks.ps1" -Unattended
if errorlevel 1 (
    echo Installation failed. Review the message above. No passwords were saved here.
    pause
    exit /b 1
)
pause

