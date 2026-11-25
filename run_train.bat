@echo off
title Training BISINDO Hybrid Ensemble (YOLOv11n-cls + Landmark MLP)
echo ========================================================
echo    Training BISINDO Hybrid Ensemble (Partisi E: Only)
echo    1. Prep Dataset (data/bisindo_cls)
echo    2. Train YOLOv11n-cls (Visual Classifier)
echo    3. Train Dual-Hand Landmark MLP (42 Keypoints)
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

echo [1/3] Menyiapkan dataset klasifikasi...
"%PYTHON_EXE%" src\dataset\prep_cls_dataset.py
if errorlevel 1 goto :err

echo.
echo [2/3] Melatih model YOLOv11n-cls...
"%PYTHON_EXE%" src\train_cls.py %*
if errorlevel 1 goto :err

echo.
echo [3/3] Melatih model Dual-Hand Landmark MLP...
"%PYTHON_EXE%" src\train_landmarks.py
if errorlevel 1 goto :err

echo.
echo ========================================================
echo   SELESAI! Kedua model sudah siap di folder weights\
echo   Jalankan run_pipeline.bat untuk uji coba webcam.
echo ========================================================
pause
exit /b 0

:err
echo [ERROR] Proses training mengalami kendala.
pause
exit /b 1
