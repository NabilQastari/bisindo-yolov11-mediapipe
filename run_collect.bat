@echo off
title BISINDO Custom Data Collector
echo ========================================================
echo   BISINDO Custom Data Collector (Kalibrasi Kamera Anda)
echo   Pilih kamera via GUI, lalu tekan a-z untuk rekam pose.
echo   Setelah selesai, jalankan run_train.bat untuk melatih!
echo ========================================================
echo.

cd /d "E:\Semester 5\C&V\Project Citra"
set "PYTHONPATH=E:\Semester 5\C&V\Project Citra;%PYTHONPATH%"
set "PYTHON_EXE=E:\Semester 5\C&V\Project Citra\venv\Scripts\python.exe"

"%PYTHON_EXE%" src\dataset\collector.py %*
pause
