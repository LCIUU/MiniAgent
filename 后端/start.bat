@echo off
chcp 65001 >nul
title AI Coding Agent Backend
cd /d %~dp0

echo ============================================
echo    AI Coding Agent Backend
echo    Close this window to stop the service
echo ============================================
echo.

where uv >nul 2>nul
if %errorlevel%==0 (
    echo [1/2] Syncing dependencies...
    uv sync
    echo [2/2] Starting backend...
    uv run python main.py
) else (
    echo [INFO] uv not found, falling back to python...
    python main.py
)

echo.
echo Service stopped.
pause
