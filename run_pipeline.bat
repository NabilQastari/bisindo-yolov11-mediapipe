@echo off
setlocal
title BISINDO Real-Time Multi-Source Translator
echo ========================================================
echo       BISINDO Real-Time Multi-Source Translator
echo       (MediaPipe 42 Keypoints + YOLOv11n-cls + Landmark MLP)
echo ========================================================
echo.

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"
set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
set "PYTHON_EXE=%ROOT%\venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [INFO] Virtual Environment belum ada di "%ROOT%\venv".
    echo        Jalankan setup_env.bat terlebih dahulu di laptop ini.
    pause
    exit /b 1
)

"%PYTHON_EXE%" src\pipeline.py %*
pause
