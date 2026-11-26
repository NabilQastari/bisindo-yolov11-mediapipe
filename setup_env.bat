@echo off
setlocal
title Setup Environment - BISINDO Translator
echo ========================================================
echo   Setup Awal Project BISINDO (Untuk Laptop Baru)
echo   - Membuat venv lokal di folder project ini
echo   - Menginstal library dari requirements.txt (--no-cache-dir)
echo   - Bobot model sudah tersedia di weights\, TIDAK perlu train ulang!
echo ========================================================
echo.

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan di PATH!
    echo         Install Python 3.10 - 3.12 terlebih dahulu dari python.org
    echo         dan centang "Add python.exe to PATH".
    pause
    exit /b 1
)

if not exist "%ROOT%\venv\Scripts\python.exe" (
    echo [1/3] Membuat Python Virtual Environment di "%ROOT%\venv"...
    python -m venv "%ROOT%\venv"
    if errorlevel 1 (
        echo [ERROR] Gagal membuat virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment sudah ada, melewati pembuatan venv.
)

echo.
echo [2/3] Menginstal dependensi CV (--no-cache-dir)...
"%ROOT%\venv\Scripts\python.exe" -m pip install --no-cache-dir -r "%ROOT%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Instalasi dependensi gagal. Periksa koneksi internet.
    pause
    exit /b 1
)

echo.
echo [3/3] Mengunci folder Ultralytics ke direktori project ini...
"%ROOT%\venv\Scripts\yolo.exe" settings "datasets_dir=%ROOT%\data" "runs_dir=%ROOT%\runs" "weights_dir=%ROOT%\weights" >nul 2>nul

echo.
echo ========================================================
echo   SETUP SELESAI!
echo   Anda TIDAK perlu training ulang.
echo   Langsung klik ganda run_pipeline.bat untuk menjalankan!
echo ========================================================
pause
