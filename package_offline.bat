@echo off
chcp 65001 > nul
title IS Systematic Review — Offline Packager

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║   IS Systematic Review  •  Offline Packager   ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ── Check Docker ──────────────────────────────────────────────────────────
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Docker is not installed.
    echo  Please install Docker Desktop to package this application.
    pause
    exit /b 1
)

:: ── Check Docker running ──────────────────────────────────────────────────
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo  [OK] Docker is running.
echo.

:: ── Build Images ─────────────────────────────────────────────────────────
echo  [>>] Building Docker images...
docker-compose build
if %errorlevel% neq 0 (
    echo  [!] Build failed. Please check the logs.
    pause
    exit /b 1
)
echo.
echo  [OK] Docker images built successfully.
echo.

:: ── Export Images ────────────────────────────────────────────────────────
echo  [>>] Exporting sr_backend:latest to sr_backend.tar...
docker save -o sr_backend.tar sr_backend:latest
if %errorlevel% neq 0 (
    echo  [!] Failed to export backend image.
    pause
    exit /b 1
)

echo  [>>] Exporting sr_frontend:latest to sr_frontend.tar...
docker save -o sr_frontend.tar sr_frontend:latest
if %errorlevel% neq 0 (
    echo  [!] Failed to export frontend image.
    pause
    exit /b 1
)

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║  Offline packaging complete!                 ║
echo  ║                                              ║
echo  ║  Created:                                    ║
echo  ║  - sr_backend.tar                            ║
echo  ║  - sr_frontend.tar                           ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo  To distribute, zip the following items:
echo  1. docker-compose.yml
echo  2. LAUNCH.bat
echo  3. STOP.bat
echo  4. sr_backend.tar
echo  5. sr_frontend.tar
echo.
pause
exit /b 0
