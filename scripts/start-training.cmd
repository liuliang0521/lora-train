@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0train-style.ps1" %*
set "train_exit=%ERRORLEVEL%"
if not "%train_exit%"=="0" echo Training command failed. Read the output above.
pause
exit /b %train_exit%
