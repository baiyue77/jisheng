@echo off
chcp 65001 >nul
setlocal
call "%~dp0media_env.cmd"
set "PY=%~dp0venv\Scripts\python.exe"
set "SCRIPT=%~dp0webui_launch.py"

if not exist "%PY%" goto :nopython
if not exist "%SCRIPT%" goto :noscript

echo.
echo   正在启动集声 ...
echo   Browser will open at http://127.0.0.1:8760
echo.
echo   Closing this window will stop the server and its child processes.
echo.
"%PY%" "%SCRIPT%" %*
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%

:nopython
echo [X] venv not found:
echo     "%PY%"
echo     See README.md for setup.
pause
exit /b 1

:noscript
echo [X] webui.py not found:
echo     "%SCRIPT%"
pause
exit /b 1
