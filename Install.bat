@echo off

if not exist "%~dp0\venv\Scripts" (
    echo Creating venv...
    python -m venv venv
)
echo Checked the venv folder. Now installing requirements...

call "%~dp0\venv\scripts\activate"

python -m pip install -U pip setuptools wheel

pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Requirements installation failed. Please remove the venv folder and run Install.bat again.
    pause
    exit /b 1
)

:: ultimatevocalremover_api uses legacy setup.py; --no-build-isolation avoids
:: isolated build environment issues on Python 3.12.
pip install --no-build-isolation -r requirements-git.txt
if errorlevel 1 (
    echo.
    echo Git requirements installation failed. Please remove the venv folder and run Install.bat again.
) else (
    echo.
    echo Requirements installed successfully.
)
pause
