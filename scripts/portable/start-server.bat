@echo off
setlocal
cd /d "%~dp0"
"%~dp0runtime\main\python.exe" -I -B -u "%~dp0launcher.py" serve %*
set "SAKURATTS_EXIT=%ERRORLEVEL%"
if not "%SAKURATTS_EXIT%"=="0" pause
exit /b %SAKURATTS_EXIT%
