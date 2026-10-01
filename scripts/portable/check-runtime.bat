@echo off
setlocal
if not exist "%~dp0runtime\main\python.exe" (
  echo Missing bundled runtime/main/python.exe. Extract the complete SakuraTTS archive again.
  echo System Python and Git are not required.
  pause
  exit /b 1
)
"%~dp0runtime\main\python.exe" -I -B -u "%~dp0launcher.py" check-runtime %*
set "SAKURATTS_EXIT=%ERRORLEVEL%"
if "%SAKURATTS_EXIT%"=="-1073741515" echo A runtime DLL is missing. Re-extract runtime/main including the bundled MSVC DLLs.
if "%SAKURATTS_EXIT%"=="-1073741701" echo Windows could not load this runtime. Use the matching Windows x64 package and check for incomplete DLL files.
pause
exit /b %SAKURATTS_EXIT%
