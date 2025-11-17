# TP: MLOps with DVC and MLflow

**Course:** M2 SID - Processus Data
**Instructor:** Feda Almuhisen
**Institution:** Aix-Marseille University
**Year:** 2025-2026

## Overview

This practical work (TP) teaches MLOps tools and practices using a real-world turbofan engine anomaly detection project. You will learn to:

- Track large datasets with **DVC** (Data Version Control)
- Track ML experiments with **MLflow**
- Version data, code, and models
- Reproduce ML workflows
- Apply MLOps best practices

## Project Structure

```
project/
├── data/
│   ├── download_data.py      # Script to download NASA dataset
│   ├── preprocessing.py       # Data preprocessing utilities
│   └── data_loader.py         # PyTorch data loaders
├── models/
│   ├── autoencoder.py         # Autoencoder model definition
│   ├── train.py               # Training script (with TODOs)
│   └── evaluate.py            # Evaluation script
├── notebooks/
│   └── 01_data_exploration.ipynb  # Data exploration notebook
├── requirements.txt           # Python dependencies
├── setup.sh                   # Automated setup script
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Quick Start

### 1. Setup Environment

**Choose your platform:**

**Windows:**
```cmd
setup.bat
```
Double-click `setup.bat` or run from Command Prompt.

**macOS macOS / Linux Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Manual setup (all platforms):**

Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

** For detailed setup instructions and troubleshooting, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

### 2. Download Dataset

```bash
python data/download_data.py
```

**Note:** If automatic download fails, manually download from:
- URL: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- Place `CMAPSSData.zip` in `data/raw/`
- Run the script again

### 3. Start Jupyter Lab

```bash
jupyter lab
```

Open `notebooks/01_data_exploration.ipynb` and follow along.

## What You'll Learn

### Part 1: Environment Setup & Data Exploration
- Setup Python environment
- Download NASA C-MAPSS turbofan dataset
- Explore data in Jupyter notebook
- Complete student exercises

### Part 2: DVC - Data Version Control
- Initialize Git and DVC
- Track datasets with DVC
- Set up remote storage
- Version data changes
- Collaborate with team members

### Part 3: MLflow - Experiment Tracking
- Complete MLflow TODOs in `models/train.py`
- Train autoencoder models
- View results in MLflow UI
- Run multiple experiments
- Compare hyperparameters

### Part 4: Model Evaluation
- Evaluate best model
- Understand anomaly detection metrics
- Calculate precision, recall, F1-score

### Part 5: Complete MLOps Workflow
- Integrate DVC + MLflow
- Reproduce experiments
- Apply best practices

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Dataset**
- 100 engines running until failure
- 21 sensor readings per cycle
- Target: Remaining Useful Life (RUL)
- Task: Anomaly detection using autoencoders

**Key Statistics:**
- Training samples: 20,631
- Test samples: 13,096
- Sensors: 21 (17 informative after preprocessing)
- Engine lifespans: 128-362 cycles

## Requirements

- Python 3.8+
- Git
- Basic ML knowledge
- Terminal/command line skills

## Deliverables

Submit the following:

1. **Answers to questions** (Q1.1 through Q5.1)
2. **Screenshots:**
   - MLflow UI with 4+ experiments
   - MLflow comparison view
   - Git log showing commits
   - Results from student exercises
3. **Best model metrics:**
   - Hyperparameters used
   - Final validation loss
   - Evaluation metrics (precision, recall, F1)
4. **Reflection** (200 words):
   - What did you learn about MLOps?
   - How would you apply this in a real project?
   - Challenges faced?

## Troubleshooting

**Module not found**
```bash
source .venv/bin/activate
```

**DVC push fails**
```bash
dvc remote list
```

**MLflow UI won't start**
```bash
mlflow ui --port 5001
```

**Jupyter kernel not found**
```bash
python -m ipykernel install --user --name turbofan
```

**Training loss is NaN**
```bash
python models/train.py --lr 0.0001
```

## Additional Resources

- MLflow Documentation: https://www.mlflow.org/docs/
- DVC Documentation: https://dvc.org/doc
- PyTorch Tutorials: https://pytorch.org/tutorials/
- NASA Dataset: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the TP instructions PDF
3. Ask me for help

---

**Good luck with your TP!**
