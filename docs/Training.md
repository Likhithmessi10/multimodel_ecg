# Model Training Pipeline & Hyperparameters

This document details the configuration, pre-processing, and execution flows used to train and optimize the ECG classification models.

---

## 1. Data Preprocessing

### 1.1 ECG Waveform Filtering
ECG waveforms are sampled at 100 Hz. They undergo zero-phase bandpass filtering from 0.5 Hz to 45 Hz using a 101-tap linear phase finite impulse response (FIR) filter to remove high-frequency noise and baseline drift.

### 1.2 Demographic Variable Processing
Patient clinical demographic metrics are imputed and scaled:
- **Missing Value Imputation**: Missing values for age, height, and weight are imputed using the median values computed *exclusively* from training fold data.
- **Normalization**: Numerical attributes (age, weight, height) are scaled to zero-mean and unit variance using training split parameters to prevent information leakage.
- **Sex Categorization**: Sex is represented as a binary variable (0 for Male, 1 for Female).

---

## 2. Optimization Settings

Training parameters are governed by `config.json` to ensure reproducible execution. Standard optimization settings include:
- **Optimizer**: AdamW optimizer.
- **Weight Decay**: Regularization with a weight decay coefficient of $10^{-4}$.
- **Learning Rate Scheduler**: PyTorch's `OneCycleLR` scheduler. This dynamically scales the learning rate from a low threshold, peaking at $2 \times \text{Initial LR}$ halfway through training, and decaying using a cosine schedule.
- **Mixed Precision (AMP)**: Leverages `torch.cuda.amp` to accelerate computations using 16-bit float formats on CUDA GPUs.
- **Gradient Clipping**: Norm clipping at a maximum value of 1.0 to prevent exploding gradients.
- **Early Stopping**: Halts execution when validation loss fails to decrease for 5 consecutive epochs.

---

## 3. Configuration Properties

The hyperparameters are managed using `config.json` in the root folder:

```json
  "training": {
    "seed": 42,
    "epochs": 15,
    "batch_size": 64,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "patience": 5
  }
```

---

## 4. Execution Command

To run standard training, execute:
```bash
python trainer.py --epochs 15 --fusion_type gated
```

To run 5-Fold Stratified Cross-Validation:
```bash
python trainer.py --run_cv --fusion_type gated
```
