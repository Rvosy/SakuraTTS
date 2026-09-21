@echo off
setlocal
"%~dp0runtime\main\python.exe" -I -B -u "%~dp0launcher.py" check-runtime
pause
