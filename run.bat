@echo off
REM Launch GliaAnalysis using the project's own virtual environment.
REM Double-click this file, or run  .\run.bat  from any shell.
REM Uses .venv\Scripts\python.exe directly -- no "activate" step and no
REM PowerShell execution-policy change needed (.bat is not subject to it).
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [GliaAnalysis] No virtual environment found at .venv
  echo Create it first, e.g.:
  echo     py -3.13 -m venv .venv
  echo     .venv\Scripts\python.exe -m pip install -r requirements-lock.txt
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" app.py %*
endlocal
