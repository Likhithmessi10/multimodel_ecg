# Experimental Protocol & Validation

This document describes the validation protocols, baseline setups, robustness studies, and statistical significance tests.

---

## 1. Validation Protocol (Patient-wise Stratified Split)

To ensure validation and prevent data leakage, PTB-XL uses a patient-wise 10-fold split:
- **Zero Leakage Split**: The dataset is split such that patient records do not span multiple folds. Folds 1-8 form the training set, fold 9 is the validation set, and fold 10 is the test set.
- **5-Fold Stratified Cross-Validation**: If `--run_cv` is specified, training executes a 5-fold cross-validation scheme where the test and validation folds shift dynamically:
  - Fold 1: Train Folds 4-10, Val Fold 3, Test Folds 1-2.
  - Fold 2: Train Folds 1-2, 6-10, Val Fold 5, Test Folds 3-4.
  - Fold 3: Train Folds 1-4, 8-10, Val Fold 7, Test Folds 5-6.
  - Fold 4: Train Folds 1-6, 10, Val Fold 9, Test Folds 7-8.
  - Fold 5: Train Folds 1-7, Val Fold 8, Test Folds 9-10.

---

## 2. Robustness Experiments Under Noise

Model robustness is evaluated on the test set across several clinical noise scenarios:
1. **Missing Metadata**: Zeroing out age, sex, weight, or height to simulate incomplete clinical records.
2. **Gaussian Noise**: Adding random Gaussian white noise $\mathcal{N}(0, \sigma^2)$ at $\sigma \in \{0.05, 0.15, 0.30\}$ to simulate electrode interference.
3. **Baseline Wander**: Injecting low-frequency sinusoidal noise (0.15 Hz) at amplitudes of 0.2 mV, 0.4 mV, and 0.6 mV to simulate patient respiration.
4. **Lead Dropout**: Randomly dropping (setting to zero) 17%, 33%, and 50% of the 12 leads to simulate detached electrodes.

---

## 3. Demographic Fairness & Bias Evaluation

Model fairness is assessed across three patient cohorts:
- **Sex**: Male vs. Female.
- **Age**: Young (age < 65) vs. Elderly (age $\ge$ 65).
- **BMI**: Normal (BMI < 25) vs. Overweight (BMI $\ge$ 25).
We report Macro F1, Micro F1, Macro AUROC, and Macro AUPRC for each cohort to evaluate potential bias.

---

## 4. Statistical Validation

To ensure empirical validity, we compute:
- **Bootstrap Confidence Intervals**: We perform 200 bootstrap iterations over the test predictions to compute 95% confidence intervals for F1 and AUROC.
- **Wilcoxon Signed-Rank Test**: A paired Wilcoxon test is run over the sample-wise Binary Cross Entropy (BCE) losses to assess performance differences between the best proposed model and baseline estimators.
- **McNemar Chi-Squared Test**: Computes the significance of classification accuracy differences between models on correct vs. incorrect predictions.
