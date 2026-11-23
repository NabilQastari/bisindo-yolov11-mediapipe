@echo off
title Training YOLOv11n BISINDO
echo ========================================================
echo         Training YOLOv11n untuk Alfabet BISINDO
echo ========================================================
echo.

cd /d "E:\Semester 5\C&V\Project Citra"
set "PYTHONPATH=E:\Semester 5\C&V\Project Citra;%PYTHONPATH%"
set "PYTHON_EXE=E:\Semester 5\C&V\Project Citra\venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual Environment tidak ditemukan di E:\Semester 5\C&V\Project Citra\venv
    pause
    exit /b 1
)

"%PYTHON_EXE%" src/train.py %*
pause
