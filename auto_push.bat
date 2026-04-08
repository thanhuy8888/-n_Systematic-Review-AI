@echo off
chcp 65001 >nul
setlocal

:: Get commit message from argument, or use default timestamp
set "msg=%*"
if "%~1"=="" (
    set "msg=Auto commit %date% %time%"
)

echo [1/3] Adding changes...
"C:\laragon\bin\git\bin\git.exe" add .

echo [2/3] Committing...
"C:\laragon\bin\git\bin\git.exe" commit -m "%msg%"

echo [3/3] Pushing to GitHub...
"C:\laragon\bin\git\bin\git.exe" push origin main

echo Done!
pause
