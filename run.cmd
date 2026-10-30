@echo off
rem ============================================================================
rem One-click launcher for the realtime translator.
rem Double-click this file. It kills any stale instance, starts the app (which
rem spawns whisper-server / llama-server itself), and opens the browser.
rem
rem Everything needed is already next to target\release\realtime-translator.exe:
rem whisper-server.exe, llama-server.exe, their runtime DLLs, and models\.
rem If you build on a different machine, copy those files the same way - see
rem README "Third-party binaries".
rem ============================================================================
cd /d %~dp0

taskkill /f /im realtime-translator.exe >nul 2>&1
taskkill /f /im llama-server.exe       >nul 2>&1
taskkill /f /im whisper-server.exe     >nul 2>&1

if not exist "target\release\realtime-translator.exe" (
    echo Build first:  cargo build --release
    pause
    exit /b 1
)

start "realtime-translator" "target\release\realtime-translator.exe"

timeout /t 3 /nobreak >nul
start "" "http://localhost:3000"

echo Started. The app runs in its own window; play audio and captions appear.
echo Stop later with:  taskkill /f /im realtime-translator.exe
exit /b 0
