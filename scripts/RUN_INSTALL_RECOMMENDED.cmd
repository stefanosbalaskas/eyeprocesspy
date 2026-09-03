@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_eyeprocesspy.ps1" -WithAllRecommended
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" (
  echo Installation did not complete successfully. Review the message above.
) else (
  echo Installation completed.
)
echo.
pause
exit /b %EXITCODE%
