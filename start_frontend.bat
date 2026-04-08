@echo off
chcp 65001 >nul
set PATH=C:\laragon\bin\nodejs\node-v18;%PATH%
echo Dang khoi dong Giao dien Web React...
cd /d c:\laragon\www\doantotnghiep\apps\web
echo Dang kiem tra npm thu vien...
call npm install
echo Dang chay giao dien Web (truy cap o cong 5173)...
call npm run dev
pause
