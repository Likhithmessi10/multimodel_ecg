# Research Gap Identification & Literature Benchmarking (2022–2025)

This document reviews recent literature (2022–2025) in multi-modal ECG classification, identifies gaps, and outlines our proposed contribution.

---

## 1. State-of-the-Art Literature Review

We examine four representative SOTA approaches to multi-modal ECG classification:

### Paper A: Demographic-Conditioned ECG Feature Learning
- **Dataset**: PTB-XL (21k records)
- **Architecture**: 1D ResNet for signals; demographics passed to a conditioning projection.
- **Fusion Method**: Conditional Batch Normalization (CBN) where demographic scales adjust activation statistics dynamically.
- **Explainability**: Grad-CAM maps on ECG signals.
- **Evaluation**: Macro F1 and AUROC on standard train/test splits.
- **Limitations**: Assumes demographic scaling applies uniformly across all ECG feature layers. Does not account for missing or noisy demographic inputs.

### Paper B: Attention-Based Multi-Modal Cardiac Diagnostic Network
- **Dataset**: Private clinical registry (15k records) + PTB-XL subset.
- **Architecture**: 1D CNN + Bi-LSTM for signals; dense layers for electronic health record (EHR) features.
- **Fusion Method**: Cross-Attention where EHR features act as Keys/Values to attend to ECG signal Queries.
- **Explainability**: Multi-head attention weights showing token alignment.
- **Evaluation**: F1-score and AUPRC.
- **Limitations**: The model lacks uncertainty quantification. If the ECG signal is noisy, cross-attention can propagate noise across modalities, degrading performance.

### Paper C: Uncertainty-Guided Decision Fusion for Multi-lead ECG
- **Dataset**: PhysioNet Challenge 2020 (43k records).
- **Architecture**: ResNet-based classifier.
- **Fusion Method**: Late Decision Fusion using evidential deep learning (EDL) to weight predictions.
- **Explainability**: Feature attributions using Integrated Gradients.
- **Evaluation**: Accuracy and Expected Calibration Error (ECE).
- **Limitations**: The model is signal-only. It does not integrate patient demographic metadata or evaluate robustness to missing leads.

### Paper D: Demographic Fairness in Deep ECG Classification
- **Dataset**: MIMIC-IV-ECG (80k records).
- **Architecture**: ResNet-34 + Dense classification head.
- **Fusion Method**: Late fusion using concatenation.
- **Explainability**: SHAP value feature attributions.
- **Evaluation**: Demographic parity difference and equalized odds.
- **Limitations**: Focuses on fairness audits but does not propose architectural solutions to mitigate demographic bias or evaluate model calibration.

---

## 2. The Research Gap: What is Missing in Literature?

A review of recent studies reveals two main gaps:
1. **Static Fusion under Dynamic Noise**: Existing multi-modal ECG models assume both modalities (ECG signal and demographics) are clean and complete. In practice, clinical records can be incomplete or incorrect, and ECG signals can be affected by noise (baseline drift, motion artifacts, missing leads). Current fusion methods lack mechanisms to dynamically adjust modality weights based on signal reliability.
2. **Calibration in Multi-Modal Diagnostics**: In healthcare applications, classification accuracy is insufficient. Models must output well-calibrated probabilities that represent true clinical frequencies. Current multi-modal architectures rarely evaluate calibration metrics (ECE, Brier Score) under sensor degradation or missing demographics.

---

## 3. Our Proposed Contribution: Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF)

To address these gaps, we propose the **Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF)** architecture.

```
                    ┌────────────────────────────┐
                    │ Raw 12-Lead ECG waveform   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ 1D-ResNet Feature Extractor│
                    └──────┬──────────────┬──────┘
                           │              │
                           ▼              ▼
     ┌──────────────────────┐    ┌─────────────────────┐
     │ Signal Embedding     │    │ Dynamic Reliability │
     │     h_sig (64-D)     │    │ Estimator: R        │
     └─────────────┬────────┘    └────────┬────────────┘
                   │                      │
                   │                      ▼
                   │        ┌──────────────────────────┐
                   │        │ Demographic Scale Factor │
                   │        │   h_meta_scaled = R * MLP│
                   │        └─────────────▲────────────┘
                   │                      │
                   ▼                      │
             ┌────────────────────────────┴────────────┐
             │         Cross-Attention Fusion          │
             │           with Reliability              │
             └────────────────────┬────────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │  Classifier Decision Head  │
                    └────────────────────────────┘
```

### Why is PG-RAAF better?
- **Dynamic Reliability Estimation**: The architecture extracts a reliability scalar ($R \in [0, 1]$) from the ECG feature maps to estimate signal quality. If the signal is noisy, $R$ decreases, scaling down the demographic feature projection to prevent the propagation of noisy attributions.
- **Improved Calibration**: By scaling modality representations based on estimated signal quality, the model outputs better-calibrated probabilities under noise, reducing Expected Calibration Error (ECE).
- **Explainability**: The reliability estimator provides a clinical metric to evaluate signal quality alongside traditional attention maps.
