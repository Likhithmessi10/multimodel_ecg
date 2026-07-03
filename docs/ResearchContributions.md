# Core Research Contributions

This document highlights the clinical and technical contributions of this research project.

---

## 1. Technical Contributions

1. **Adaptive Multi-Modal Fusion**: We implement and evaluate five distinct late-fusion strategies: Concatenation, Gated, Cross-Attention, Feature Attention, and Dynamic Weighted Fusion. This replaces simple concatenation, allowing the model to learn relationships between ECG signals and patient demographics.
2. **Uncertainty Quantification (MC Dropout)**: We implement Monte Carlo Dropout to generate epistemic uncertainty scores for each diagnostic prediction, allowing the model to flag low-confidence outputs.
3. **Reproducible Preprocessing with Zero Leakage**: Preprocessing ensures that missing value imputation and feature scaling parameters are fitted *only* on training data, preventing validation information leakage.
4. **Demographic Bias Assessment**: The framework includes fairness evaluation modules to profile performance parity across sex, age, and BMI cohorts.

---

## 2. Clinical Significance

1. **Safety-Critical Decision Support**: By flagging uncertain predictions ($H > 0.75$), the system reduces automated misdiagnoses, notifying medical staff when manual ECG inspection is required.
2. **Robustness under Real-World Noise**: The robustness experiments evaluate performance under clinical signal degradation, such as electrode drift, baseline wander, and detached leads, simulating real-world hospital environments.
3. **Transparent Decision Making (Explainable AI)**: By combining Grad-CAM (temporal attention), Integrated Gradients (lead contribution), and SHAP (demographic attribution), the framework provides explanations for its predictions, helping clinicians verify diagnostic decisions.
