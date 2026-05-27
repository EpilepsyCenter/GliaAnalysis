@echo off
title GliaAnalysis (native window)
cd /d "%~dp0"
echo ============================================================
echo   GliaAnalysis  ^|  native desktop app (WebView2)
echo   The app opens in its own window in a few seconds.
echo   Close the app window (or this console) to stop the server.
echo ============================================================
echo.
".venv\Scripts\python.exe" -m glia_dash.desktop
echo.
echo Server stopped.
pause
