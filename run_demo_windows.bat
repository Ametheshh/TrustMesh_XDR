@echo off
setlocal
cd /d "%~dp0"

echo ============================================================================
echo   TrustMesh XDR - Windows Automated Setup and Mentor Demonstration Runner
echo ============================================================================
echo.

set "PY_CMD="
python --version >nul 2>nul
if not errorlevel 1 set "PY_CMD=python"
if defined PY_CMD goto :found_python

py -3 --version >nul 2>nul
if not errorlevel 1 set "PY_CMD=py -3"
if defined PY_CMD goto :found_python

goto :no_python

:found_python
echo [*] Using Python command: %PY_CMD%

if exist ".venv\Scripts\python.exe" goto :venv_ready
echo [*] Creating local virtual environment in .venv ...
%PY_CMD% -m venv .venv

:venv_ready
set "RUN_PY=%PY_CMD%"
if exist ".venv\Scripts\python.exe" set "RUN_PY=.venv\Scripts\python.exe"

echo [*] Installing required packages: numpy and matplotlib ...
%RUN_PY% -m pip install --upgrade pip --quiet
%RUN_PY% -m pip install numpy matplotlib --quiet
if not errorlevel 1 goto :run_bench

echo [!] Retrying package installation with --user flag ...
%PY_CMD% -m pip install --user numpy matplotlib
set "RUN_PY=%PY_CMD%"

:run_bench
echo.
echo [*] Running benchmark across 262,879 real network flows ...
%RUN_PY% scripts\run_model_comparison_report.py --check
if errorlevel 1 goto :bench_error

if exist "reports\model_comparison_report.png" start "" "reports\model_comparison_report.png"

echo.
echo [DONE] Demonstration complete! Show the CLI output above and the opened graph to your mentors.
pause
exit /b 0

:no_python
echo [ERROR] Python 3 was not found on this Windows laptop.
echo Please install Python 3 from https://www.python.org/downloads/ and check Add Python to PATH.
pause
exit /b 1

:bench_error
echo [ERROR] Benchmark script encountered an error.
pause
exit /b 1
