@echo off
title BISINDO Custom Data Collector
echo ========================================================
echo   BISINDO Custom Data Collector (Kalibrasi Kamera Anda)
echo   Tekan tombol a-z untuk merekam pose tangan Anda.
echo   Setelah selesai, jalankan run_train.bat untuk melatih!
echo ========================================================
echo.

cd /d "E:\Semester 5\C&V\Project Citra"
set "PYTHONPATH=E:\Semester 5\C&V\Project Citra;%PYTHONPATH%"
set "PYTHON_EXE=E:\Semester 5\C&V\Project Citra\venv\Scripts\python.exe"

set /p CAM_ID="Masukkan indeks kamera (default 0, misal 0 atau 3): "
if "%CAM_ID%"=="" set "CAM_ID=0"

"%PYTHON_EXE%" src\dataset\collector.py --cam %CAM_ID%
pause
