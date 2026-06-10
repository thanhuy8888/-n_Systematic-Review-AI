@echo off
chcp 65001 >nul
echo Dang chay danh gia screening (leakage-free)... Vui long cho trong giay lat...
call .\venv\Scripts\python.exe experiments\baselines\evaluate_screening_clean.py
echo.
echo Da hoan thanh! Copy khoi "COPY THESE INTO TABLE 5" o tren de dua vao bao cao.
pause
