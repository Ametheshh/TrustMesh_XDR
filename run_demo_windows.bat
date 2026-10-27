@echo off
setlocal
cd /d "%~dp0"

echo ============================================================================
echo   TrustMesh XDR - Windows Automated Setup ^& Mentor Demonstration Runner
echo ============================================================================
echo.

:: 1. Locate Python on Windows (checks 'python' first, then 'py -3')
set "PY_CMD="
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set "PY_CMD=py -3"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python 3 was not found on this Windows laptop.
    echo Please install Python 3 from https://www.python.org/downloads/ and check "Add Python to PATH".
    pause
    exit /b 1
)

echo [*] Using Python command: %PY_CMD%

:: 2. Create/use local virtual environment if possible, otherwise install to user site
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating local virtual environment (.venv)...
    %PY_CMD% -m venv .venv
)

if exist ".venv\Scripts\python.exe" (
    set "RUN_PY=.venv\Scripts\python.exe"
) else (
    set "RUN_PY=%PY_CMD%"
)

:: 3. Install required packages (numpy, matplotlib)
echo [*] Installing required packages (numpy, matplotlib)...
%RUN_PY% -m pip install --upgrade pip --quiet
%RUN_PY% -m pip install numpy matplotlib --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [!] Retrying package installation with --user flag...
    %PY_CMD% -m pip install --user numpy matplotlib
    set "RUN_PY=%PY_CMD%"
)

:: 4. Run the full 18 Small Models vs. Huge Federated Model benchmark + self-check
echo.
echo [*] Running benchmark across 262,879 real network flows...
%RUN_PY% scripts\run_model_comparison_report.py --check
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Benchmark script encountered an error.
    pause
    exit /b 1
)

:: 5. Automatically open the generated comparison graph image on Windows
if exist "reports\model_comparison_report.png" (
    echo [*] Opening visual comparison report (reports\model_comparison_report.png)...
    start "" "reports\model_comparison_report.png"
)

echo.
echo [DONE] Demonstration complete! You can show the CLI output above and the opened graph to your mentors.
pause
