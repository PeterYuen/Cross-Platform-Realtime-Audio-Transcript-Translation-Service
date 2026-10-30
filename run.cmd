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

rem Keep one generation of history so the log never grows without bound.
if exist server_console.log (
    del /q server_console.log.old 2>nul
    ren server_console.log server_console.log.old
)

set RUST_LOG=info
start "realtime-translator" cmd /c "target\release\realtime-translator.exe >> server_console.log 2>&1"

timeout /t 3 /nobreak >nul
start "" "http://localhost:3000"

echo Started. Logs: server_console.log in this folder. Play audio and captions appear.
echo Stop later with:  taskkill /f /im realtime-translator.exe
exit /b 0
