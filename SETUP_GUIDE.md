# Setup Guide - Cross-Platform

This guide provides setup instructions for **Windows**, **macOS**, and **Linux**.

---

## Quick Start by Platform

### Windows

**Option 1: Automated (Recommended)**
```cmd
setup.bat
```
Double-click `setup.bat` or run it from Command Prompt.

**Option 2: Manual**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

**Option 1: Automated (Recommended)**
```bash
chmod +x setup.sh
./setup.sh
```

**Option 2: Manual**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Prerequisites

### All Platforms
- Python 3.8 or higher
- Git (for Part 2 of the TP)
- Internet connection (for downloads)
- ~500 MB free disk space

### Platform-Specific

#### Windows
- **Python Installation:**
  - Download from: https://www.python.org/downloads/
  - IMPORTANT: **IMPORTANT:** Check "Add Python to PATH" during installation
  - Verify: `python --version` in Command Prompt

- **Git Installation:**
  - Download from: https://git-scm.com/download/win
  - Use default settings

#### macOS
- **Python Installation:**
  - Often pre-installed (check version: `python3 --version`)
  - If needed: `brew install python3` (requires Homebrew)
  - Or download from: https://www.python.org/downloads/

- **Git Installation:**
  - Often pre-installed
  - If needed: `brew install git`
  - Or install Xcode Command Line Tools: `xcode-select --install`

#### Linux (Ubuntu/Debian)
- **Python Installation:**
  ```bash
  sudo apt update
  sudo apt install python3 python3-venv python3-pip
  ```

- **Git Installation:**
  ```bash
  sudo apt install git
  ```

---

## Step-by-Step Setup

### Step 1: Extract Package

**Windows:**
- Right-click `turbofan_tp_student_package.zip`
- Select "Extract All..."
- Navigate to extracted `student-template` folder

**macOS/Linux:**
```bash
unzip turbofan_tp_student_package.zip
cd student-template
```

### Step 2: Run Setup Script

**Windows:**
```cmd
setup.bat
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**What the script does:**
1. Checks Python version
2. Creates virtual environment (`.venv`)
3. Installs all dependencies from `requirements.txt`
4. Verifies installation
5. Checks for dataset

### Step 3: Activate Environment

You need to activate the virtual environment every time you start a new terminal session.

**Windows:**
```cmd
.venv\Scripts\activate.bat
```
Prompt changes to: `(.venv) C:\path\to\student-template>`

**macOS/Linux:**
```bash
source .venv/bin/activate
```
Prompt changes to: `(.venv) username@machine:~/student-template$`

### Step 4: Download Dataset

```bash
python data/download_data.py
```

**If automatic download fails:**
1. Go to: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
2. Download `CMAPSSData.zip`
3. Place it in `data/raw/` folder
4. Run the script again

### Step 5: Start Jupyter

**All Platforms:**
```bash
jupyter lab
```

Or if using VS Code:
1. Open `student-template` folder in VS Code
2. Install Python extension
3. Open a `.ipynb` file
4. Select kernel: Choose `.venv` Python

---

## Verification

After setup, verify everything works:

**Test Python packages:**
```bash
python -c "import torch, mlflow, pandas, jupyter; print('All packages OK!')"
```

**Check directory structure:**
```
student-template/
├── data/
│   ├── raw/           (should contain CMAPSSData.zip after download)
│   └── processed/
├── models/
├── notebooks/
└── .venv/             (created by setup script)
```

---

## IDE Setup

### VS Code (Recommended)

1. **Install VS Code:**
   - Download: https://code.visualstudio.com/

2. **Install Extensions:**
   - Python (Microsoft)
   - Jupyter (Microsoft)
   - Pylance (Microsoft)

3. **Open Project:**
   - File → Open Folder → Select `student-template`

4. **Select Interpreter:**
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
   - Type: "Python: Select Interpreter"
   - Choose: `.venv/bin/python` or `.venv\Scripts\python.exe`

5. **Open Notebook:**
   - Navigate to `notebooks/01_data_exploration.ipynb`
   - Click "Select Kernel" → Choose `.venv`

### PyCharm

1. Open `student-template` as a project
2. Settings → Project → Python Interpreter
3. Add Interpreter → Existing Environment
4. Select: `.venv/bin/python` or `.venv\Scripts\python.exe`

### Jupyter Lab (Browser-based)

```bash
# Activate environment first
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate.bat   # Windows

# Start Jupyter Lab
jupyter lab
```

Opens in browser at: http://localhost:8888

---

## Common Issues

### Issue: "python: command not found" (Windows)

**Solution:** Python not in PATH
1. Reinstall Python from python.org
2. Check "Add Python to PATH"
3. Restart Command Prompt

### Issue: "python3: command not found" (macOS/Linux)

**Solution:** Try `python` instead of `python3`, or install:
```bash
# macOS
brew install python3

# Ubuntu/Debian
sudo apt install python3
```

### Issue: Permission denied (macOS/Linux)

**Solution:** Make script executable
```bash
chmod +x setup.sh
```

### Issue: Execution policy error (Windows PowerShell)

**Solution:** Use Command Prompt (CMD) instead of PowerShell, or:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Issue: SSL certificate error during pip install

**Solution:** Update pip and try again
```bash
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### Issue: PyTorch installation fails

**Solution:** Install CPU-only version explicitly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Issue: Jupyter kernel not found

**Solution:** Install kernel manually:
```bash
python -m ipykernel install --user --name turbofan --display-name "Python (turbofan)"
```

### Issue: Port 5000 already in use (MLflow)

**Solution:** Use different port:
```bash
mlflow ui --port 5001
```

---

## Alternative: Using Conda (Optional)

If you prefer Conda over venv:

```bash
# Create environment
conda create -n turbofan python=3.10

# Activate
conda activate turbofan

# Install packages
pip install -r requirements.txt
```

---

## Getting Help

1. **Check this guide first**
2. **Review README.md** in the project root
3. **Check TP_Instructions.tex/pdf** for detailed information
4. **Ask me for help**

---

## Platform-Specific Commands Reference

| Task | Windows | macOS/Linux |
|------|---------|-------------|
| Run setup | `setup.bat` | `./setup.sh` |
| Activate venv | `.venv\Scripts\activate.bat` | `source .venv/bin/activate` |
| Deactivate venv | `deactivate` | `deactivate` |
| Python command | `python` | `python3` or `python` |
| Pip command | `pip` | `pip3` or `pip` |
| Path separator | `\` (backslash) | `/` (forward slash) |
| List files | `dir` | `ls` |
| Clear screen | `cls` | `clear` |

---

**Setup complete! Proceed to Part 1 of the TP.** OK
