@echo off
chcp 65001 >nul
setlocal

REM Thin wrapper. Real logic is in setup_diarize.py
REM (batch + nested blocks + CJK text is too fragile).
set "PY=%~dp0venv\Scripts\python.exe"
if exist "%PY%" (
    "%PY%" "%~dp0setup_diarize.py" %*
) else (
    py -3 "%~dp0setup_diarize.py" %*
)
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" echo [X] setup failed, exit code = %RC%
pause
exit /b %RC%
