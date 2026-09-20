@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "SAKURATTS_PYTHON=python"
if exist ".venv-windows-runtime\Scripts\python.exe" set "SAKURATTS_PYTHON=.venv-windows-runtime\Scripts\python.exe"
"%SAKURATTS_PYTHON%" -u -m sakuratts serve %*
set "SAKURATTS_EXIT=%ERRORLEVEL%"
if not "%SAKURATTS_EXIT%"=="0" pause
exit /b %SAKURATTS_EXIT%
