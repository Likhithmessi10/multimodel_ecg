# Peer Review Report (IEEE Transactions Style)

**Manuscript Title**: Multi-Modal Adaptive Late-Fusion Network for Cardiac Diagnostics on 12-Lead ECG Signals and Patient Demographics  
**Reviewer Recommendation**: Major Revision

---

## 1. Existing Strengths
- **Multi-Modal Integration**: The project correctly identifies a clinically important task—fusing raw 12-lead time-series waveforms with patient-specific demographics (age, sex, height, weight).
- **Comprehensive Baselines**: The study compares XGBoost (demographics only) with multiple deep time-series neural networks (1D CNN, InceptionTime, ResNet, Transformer) and various late-fusion models.
- **Explainability Suite**: The integration of Grad-CAM (temporal attention), Integrated Gradients (lead contribution), and Exact Shapley Values (demographic features) addresses the "black-box" validation bottleneck.
- **Robustness Swings**: The code runs automated evaluation loops simulating clinical noise (Gaussian noise, baseline wander, lead dropout, missing metadata).
- **Zero-Leakage Preprocessing**: Preprocessing correctly fits metadata scaling and imputation params *exclusively* on the training splits.

---

## 2. Existing Weaknesses
- **Lack of True Architectural Novelty**: While the code implements several fusion mechanisms (Gated, Cross-Attention, Feature Attention, Dynamic Weighting), these are standard, generic fusion techniques. The paper lacks a single, cohesive, novel architecture designed specifically for clinical/physiological constraints.
- **Absent Calibration Metrics**: Medical AI models must be calibrated. The paper reports probabilities but fails to compute the **Expected Calibration Error (ECE)** or the **Brier Score**, which are critical metrics for clinical safety.
- **Superficial Clinical Validation**: The analysis lacks disease-specific clinical interpretation. It does not explain *why* specific leads or demographic features contribute to specific diagnoses (e.g., mapping Lead II/III/aVF to inferior Myocardial Infarction).
- **Limited Robustness Scenarios**: The robustness suite evaluates missing metadata but does *not* evaluate **incorrect metadata** (e.g., wrong age, wrong sex), which represents a common real-world clinical data entry error.
- **Limited Generalizability**: The model is evaluated solely on the PTB-XL dataset. It lacks cross-dataset evaluation or leave-group-out validation to assess generalizability across hospital settings.

---

## 3. Missing Experiments
- **Incorrect Demographic Robustness**: Sweeps that simulate patient chart mismatches by adding noise or swapping age/sex values to evaluate prediction stability.
- **Detailed Calibration Profiling**: Analysis of Brier score and ECE across different fusion strategies to evaluate whether adaptive fusion improves calibration.
- **Dataset Size Ablations**: Training with varying subsets ($N \in \{100, 500, 1000, \text{all}\}$) to evaluate data efficiency under low-resource medical conditions.

---

## 4. Missing Comparisons
- **Late Fusion vs. Joint/Early Fusion**: The framework only evaluates late-fusion setups. It lacks comparison with joint multi-modal representation learning (e.g., unified transformers).
- **Model Efficiency Benchmarking**: Systematic comparison of model size, parameter counts, FLOPs, and inference speeds across all models.

---

## 5. Missing Baselines
- **Early Fusion Baseline**: Fusing demographics at the input level (e.g., tiling demographic values as extra input channels on the ECG raw matrix).
- **Advanced Multi-Modal Baselines**: Unified cross-attention networks (e.g., Perceiver-like models) representing the state-of-the-art.

---

## 6. Missing Statistical Analysis
- **Statistically Bound Metrics**: Test F1/AUROC metrics are reported as flat scalars. A publication-grade paper requires reporting mean $\pm$ standard deviation across cross-validation folds and statistical significance tests.
- **Pairwise Wilcoxon Tests**: Statistical comparisons across all baseline networks, not just XGBoost vs. Proposed.

---

## 7. Missing Explainability
- **Class-wise Lead Saliency**: Explanations that distinguish how lead importance shifts across different cardiac conditions (e.g., NORM vs. MI vs. HYP).
- **Global Saliency Attributions**: Aggregate SHAP/feature attributions compiled across the entire test set, rather than a single local patient case.

---

## 8. Missing Clinical Validation
- **Physiological Mapping**: Evaluating whether the model's visual attention aligns with established clinical criteria (e.g., checking if the model focuses on the ST-segment during an MI prediction).
- **Expert Alignment**: Evaluation of agreement between model explainability maps and clinical guidelines.

---

## 9. Missing Robustness
- **Demographic Chart Mismatch**: Swapping demographic cards (e.g., feeding a female's ECG with a male's age/sex profile) to evaluate modality reliability.
- **Dynamic Noise Scenarios**: Simulating muscle artifacts and electrode movement noise at varying signal-to-noise ratios (SNR).

---

## 10. Missing Reproducibility
- **Model Configuration Registry**: Exposing configuration files that allow researchers to reproduce the exact model weights.
- **Environment Containment**: A reproducible environment specification (e.g., Docker container) for the training pipeline.

---

## 11. Missing Novelty
- **Standard Fusion Modules**: Generic gated/attention blocks from general computer vision are used. The study needs to propose a novel, customized architecture—such as **Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF)**—to justify publication in a top-tier medical AI venue.

---

## 12. Reviewer Criticisms & Action Items
1. *"The authors present a variety of standard fusion techniques but do not propose a single, cohesive, physiologically motivated architecture."*
   - **Action**: Design and implement a novel **Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF)** model that dynamically scales demographic embeddings based on ECG-derived signal reliability.
2. *"For medical applications, classification accuracy is insufficient. The authors must report calibration statistics (ECE and Brier score)."*
   - **Action**: Implement and log Expected Calibration Error (ECE) and Brier Score across all models, exporting reliability plots.
3. *"The robustness study is interesting but incomplete. How does the model perform under incorrect metadata entries?"*
   - **Action**: Implement robustness evaluations simulating demographic chart mismatch and incorrect demographic inputs.
4. *"The explainability results are purely qualitative for a single patient record. The authors should evaluate class-wise lead contributions across the test set."*
   - **Action**: Automate global lead-wise and cohort-wise feature attribution profiling across the test dataset.
