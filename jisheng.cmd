@echo off
chcp 65001 >nul
setlocal
call "%~dp0media_env.cmd"
set "PY=%~dp0venv\Scripts\python.exe"
set "SCRIPT=%~dp0jisheng.py"

if not exist "%PY%" goto :nopython
if not exist "%SCRIPT%" goto :noscript

if "%~1"=="" goto :usage

"%PY%" "%SCRIPT%" %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [X] exit code = %RC%
if exist "%~1" pause
exit /b %RC%

:usage
echo.
echo   集声
echo.
echo   FIRST TIME
echo     jisheng.cmd --download      download the model (~1.4 GB)
echo     jisheng.cmd --check         verify environment
echo.
echo   THEN
echo     [1] Drag a video file onto this jisheng.cmd  = single episode
echo     [2] Drag a season FOLDER onto it         = whole season
echo     [3] Or run:  jisheng.cmd "C:\path\EP01.mkv"
echo.
echo   SPEAKER LABELS (add [speaker N] to every line)
echo     jisheng.cmd "C:\path\EP01.mkv" --diarize
echo     one-time setup: run setup_diarize.cmd, then set HF_TOKEN
echo.
echo   Output goes to  the project's output folder.
echo   Full options below.
echo.
"%PY%" "%SCRIPT%" --help
pause
exit /b 1

:nopython
echo [X] Python not found:
echo     "%PY%"
echo     See README.md for setup.
pause
exit /b 1

:noscript
echo [X] Script not found:
echo     "%SCRIPT%"
pause
exit /b 1
