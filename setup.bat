@echo off
REM Setup Script for Turbofan MLOps TP (Windows)
REM Run this script to set up your environment

echo ========================================
echo Turbofan MLOps TP - Setup (Windows)
echo ========================================
echo.

REM Check Python version
echo Step 1: Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8+ from python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
python --version
echo OK Python found
echo.

REM Create virtual environment
echo Step 2: Creating virtual environment...
if exist ".venv" (
    echo Virtual environment already exists. Skipping...
) else (
    python -m venv .venv
    echo OK Virtual environment created
)
echo.

REM Activate and install requirements
echo Step 3: Installing requirements...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo OK Requirements installed
echo.

REM Verify key packages
echo Step 4: Verifying installation...
python -c "import torch, mlflow, pandas, jupyter; print('OK All packages installed successfully!')"
if %errorlevel% neq 0 (
    echo ERROR: Some packages failed to install
    pause
    exit /b 1
)
echo.

REM Check data
echo Step 5: Checking for data...
if exist "data\raw\CMAPSSData.zip" (
    echo OK Data found!
) else (
    echo WARNING: Data not found.
    echo Please run: python data/download_data.py
)
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Activate environment: .venv\Scripts\activate.bat
echo 2. Download data: python data/download_data.py
echo 3. Start Jupyter Lab: jupyter lab
echo.
echo Or open in VS Code and select kernel!
echo.
pause
