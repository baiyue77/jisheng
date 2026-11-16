@echo off
rem Configure only this launcher process; never change the system PATH.
if exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" set "PATH=%~dp0tools\ffmpeg\bin;%PATH%"
if defined FFMPEG_DIR if exist "%FFMPEG_DIR%\ffmpeg.exe" set "PATH=%FFMPEG_DIR%;%PATH%"
