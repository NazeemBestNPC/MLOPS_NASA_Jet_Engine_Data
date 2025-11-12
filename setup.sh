#!/bin/bash
#  Setup Script for Turbofan MLOps TP
# Run this script to set up your environment

echo "========================================"
echo "Turbofan MLOps TP - Setup"
echo "========================================"
echo ""

# Check Python version
echo "Step 1: Checking Python version..."
python --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3 not found. Please install Python 3.8+."
    exit 1
fi
echo "OK Python found"
echo ""

# Create virtual environment
echo "Step 2: Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    python -m venv .venv
    echo "OK Virtual environment created"
fi
echo ""

# Activate and install requirements
echo "Step 3: Installing requirements..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "OK Requirements installed"
echo ""

# Verify key packages
echo "Step 4: Verifying installation..."
python << EOF
try:
    import torch
    import mlflow
    import pandas
    import jupyter
    print("OK All packages installed successfully!")
except ImportError as e:
    print(f"ERROR: Missing package: {e}")
    exit(1)
EOF
echo ""

# Check data
echo "Step 5: Checking for data..."
if [ -f "data/raw/CMAPSSData.zip" ]; then
    echo "OK Data found!"
else
    echo "WARNING: Data not found."
    echo "Please run: python data/download_data.py"
fi
echo ""

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Download data: python data/download_data.py"
echo "3. Start Jupyter Lab: jupyter lab"
echo ""
echo "Or open in VS Code and select kernel!"
echo ""