@echo off
chcp 65001 >nul
echo Dang chay danh gia AI (Deep Learning Transformer)... Vui long cho trong giay lat...
call .\venv\Scripts\python.exe experiments\baselines\evaluate_transformer.py
echo.
echo Da hoan thanh! Ban co the copy ket qua tren de dua vao bao cao.
pause
