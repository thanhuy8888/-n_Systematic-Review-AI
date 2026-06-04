@echo off
chcp 65001 > nul
title IS Systematic Review — Stop

echo.
echo  [..] Stopping IS Systematic Review...
docker-compose down
echo.
echo  [OK] Application stopped. Your data is saved.
echo.
pause
