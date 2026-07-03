# Multi-Modal Late-Fusion ECG Diagnostics on PTB-XL

This repository contains a multi-modal classification network for cardiac diagnosis on the **PTB-XL** ECG dataset. The framework leverages a late-fusion scheme combining time-series waveform feature extraction from a **1D-ResNet CNN** and clinical metadata embeddings from an **MLP**.

---

## Key Features & Refinements

### 1. Multi-Label Classification Framework
Clinical cardiovascular diagnostics require supporting concurrent diagnoses (e.g., active Myocardial Infarction and Conduction Disturbance). 
* **Activation**: The networks utilize an explicit `nn.Sigmoid()` activation layer at the output classification head to compute independent probabilities.
* **Loss**: Trained with a custom numerically stable wrapper subclass of `nn.BCEWithLogitsLoss` that handles probability predictions directly, preventing double-sigmoid gradient distortions while maintaining full backpropagation accuracy.

### 2. Zero-Leakage Tabular Data Preprocessing
Pre-fitting scalers over the entire metadata causes leakage of test/validation split stats (means/variances) into training parameters.
* **Fix**: Preprocessing isolates Folds 1–8 (Training) for the `.fit_transform()` phase. Folds 9 (Validation) and 10 (Testing) are mapped strictly through the `.transform()` function using the training fold parameters.

### 3. Isolated Baseline Benchmarks
Reviewers require comparative benchmarking to verify that the multimodal network utilizes both modalities effectively. The pipeline trains and compares three configurations:
* **Tabular-Only Baseline**: XGBoost MultiOutput classifier operating strictly on clinical demographics (age, sex, height, weight).
* **Signal-Only Baseline**: 1D-ResNet CNN classifier operating strictly on the 12-lead time-series waveforms.
* **Proposed Late-Fusion Model**: The full integrated network fusing both embeddings at the final classifier junction.

---

## Directory Structure

```
researchpaper/
├── data/                      # Contains raw datasets and cache structures
│   └── cache/                 # Preprocessed/scaled arrays and metadata scalers
├── models/                    # Saved checkpoints (.pth and .joblib)
├── paper_artifacts/           # Generated ablation tables, stats, and plots
│   ├── README.md              # Artifact description
│   ├── figure1_loss_curves.png
│   ├── figure2_roc_curves.png
│   ├── figure3_pr_curves.png
│   ├── table1_hyperparameters.csv
│   ├── table2_ablation_results.csv
│   └── manuscript_evidence_deck.txt  # Comprehensive parameter and metrics log
├── app.py                     # Streamlit clinical dashboard UI
├── download_data.py           # Automated dataset downloader utility
├── multimodal_fusion_net.py   # PyTorch Model architectures
├── requirements.txt           # Python environment packages
├── trainer.py                 # Core training loop and execution engine
├── start.sh                   # Auto-setup and launch shell script (Mac/Linux)
└── start.bat                  # Auto-setup and launch batch script (Windows)
```

---

## Local Setup & Quick Start

Automated startup scripts are provided to configure a virtual environment, install requirements, download records, train the network, and run the Streamlit dashboard locally.

### Windows Laptop (Command Prompt or PowerShell)
Double-click or run the batch script from the repository root:
```cmd
start.bat
```

### Mac / Linux Terminal
Grant executable permissions and execute the shell script:
```bash
chmod +x start.sh
./start.sh
```

---

## Manual Execution Commands

If you prefer to run the steps individually, activate your environment and execute:

### 1. Download Dataset (Light Mode: 1000 records)
```bash
python download_data.py --num_records 1000
```
*Note: To download the full 1.84 GB PTB-XL dataset, pass the `--full` argument.*

### 2. Train Models (Sequential Sweep or Isolated)
You can train the entire sweep sequentially:
```bash
python trainer.py --epochs 15
```
Or isolate execution to a specific branch (loading other checkpoints from disk to construct the ablation tables):
```bash
python trainer.py --model_type tabular
python trainer.py --model_type signal
python trainer.py --model_type fusion
```

### 3. Run Web App Dashboard
```bash
streamlit run app.py
```
