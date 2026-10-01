@echo off
setlocal
if not exist "%~dp0runtime\main\python.exe" (
  echo Missing bundled runtime/main/python.exe. Extract the complete SakuraTTS archive again.
  echo System Python and Git are not required.
  pause
  exit /b 1
)
cd /d "%~dp0"
"%~dp0runtime\main\python.exe" -I -B -u "%~dp0launcher.py" serve %*
set "SAKURATTS_EXIT=%ERRORLEVEL%"
if not "%SAKURATTS_EXIT%"=="0" pause
exit /b %SAKURATTS_EXIT%
