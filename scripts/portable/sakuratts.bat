@echo off
setlocal
if not exist "%~dp0runtime\main\python.exe" (
  echo Missing bundled runtime/main/python.exe. Extract the complete SakuraTTS archive again.
  echo System Python and Git are not required.
  exit /b 1
)
"%~dp0runtime\main\python.exe" -I -B -u "%~dp0launcher.py" %*
exit /b %ERRORLEVEL%
