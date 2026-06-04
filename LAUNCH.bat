@echo off
chcp 65001 > nul
title IS Systematic Review — Launcher

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║   IS Systematic Review  •  AI Assistant      ║
echo  ║   Vietnam National University — Hanoi        ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ── Check Docker ──────────────────────────────────────────────────────────
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Docker Desktop is not installed.
    echo.
    echo  Please download and install Docker Desktop:
    echo  https://www.docker.com/products/docker-desktop/
    echo.
    echo  After installing, restart this file.
    pause
    start https://www.docker.com/products/docker-desktop/
    exit /b 1
)

:: ── Check Docker is running ───────────────────────────────────────────────
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Docker Desktop is installed but not running.
    echo  Please start Docker Desktop and wait for it to be ready, then run this file again.
    pause
    exit /b 1
)

echo  [OK] Docker is running.
echo.

:: ── Check for offline package files ────────────────────────────────────────
set "OFFLINE_MODE=0"
if exist "sr_backend.tar" if exist "sr_frontend.tar" (
    set "OFFLINE_MODE=1"
    echo  [>>] Offline installer package detected.
)

if "%OFFLINE_MODE%"=="1" (
    :: Check if images are already imported
    docker image inspect sr_backend:latest >nul 2>&1
    set "HAS_BACKEND_IMG=%errorlevel%"
    
    docker image inspect sr_frontend:latest >nul 2>&1
    set "HAS_FRONTEND_IMG=%errorlevel%"
    
    if %HAS_BACKEND_IMG% neq 0 (
        echo  [>>] Loading offline Docker Backend image (sr_backend.tar)...
        docker load -i sr_backend.tar
    ) else (
        echo  [OK] Docker Backend image already loaded.
    )
    
    if %HAS_FRONTEND_IMG% neq 0 (
        echo  [>>] Loading offline Docker Frontend image (sr_frontend.tar)...
        docker load -i sr_frontend.tar
    ) else (
        echo  [OK] Docker Frontend image already loaded.
    )
    echo.
)

:: ── Build and start ───────────────────────────────────────────────────────
echo  [>>] Starting the application...
if "%OFFLINE_MODE%"=="1" (
    docker-compose up -d
) else (
    echo      (First run downloads AI models ~1 GB — may take 10-20 minutes)
    echo      (Subsequent runs start in under 30 seconds)
    echo.
    docker-compose up --build -d
)

if %errorlevel% neq 0 (
    echo.
    echo  [!] Something went wrong. Check the logs:
    echo      docker-compose logs
    pause
    exit /b 1
)

:: ── Wait for frontend to be ready ─────────────────────────────────────────
echo.
echo  [..] Waiting for the application to be ready...
:wait_loop
timeout /t 3 /nobreak > nul
docker-compose ps | findstr "sr_frontend" | findstr "Up" > nul
if %errorlevel% neq 0 goto wait_loop

:: ── Open browser ──────────────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║  Application is ready!                       ║
echo  ║                                              ║
echo  ║  Opening: http://localhost                   ║
echo  ║                                              ║
echo  ║  To stop: run STOP.bat                       ║
echo  ╚══════════════════════════════════════════════╝
echo.

timeout /t 2 /nobreak > nul
start http://localhost

exit /b 0
