@echo off
setlocal
title BISINDO Custom Data Collector
echo ========================================================
echo   BISINDO Custom Data Collector (Kalibrasi Kamera Anda)
echo   Pilih kamera via GUI, lalu tekan a-z untuk rekam pose.
echo   Setelah selesai, jalankan run_train.bat untuk melatih!
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

"%PYTHON_EXE%" src\dataset\collector.py %*
pause
