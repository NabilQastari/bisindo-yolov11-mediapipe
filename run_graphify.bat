@echo off
setlocal
echo ========================================================
echo   Menjalankan Graphify untuk Obsidian Vault 'brain'
echo ========================================================
echo.

cd /d "%~dp0"

:: Gunakan Virtual Environment lokal di Drive E
if exist "%~dp0venv\Scripts\graphify.exe" (
    echo [1/3] Menggunakan Python Virtual Environment di Drive E...
    set "GRAPHIFY_CMD=%~dp0venv\Scripts\graphify.exe"
) else (
    echo [1/3] Memeriksa instalasi Graphify...
    set "GRAPHIFY_CMD=graphify"
)

echo.
echo [2/3] Memetakan knowledge graph dari folder 'brain'...
"%GRAPHIFY_CMD%" update brain

echo.
echo [3/3] Menyinkronkan file visualisasi ke folder 'graphify'...
if exist "%~dp0brain\graphify-out" (
    xcopy /Y /Q "%~dp0brain\graphify-out\*.*" "%~dp0graphify\" >nul 2>nul
)

echo.
echo ========================================================
echo Selesai! Buka file 'graphify\graph.html' di browser Anda.
echo ========================================================
pause
