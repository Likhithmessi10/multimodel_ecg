# IEEE Conference Manuscript Template (Max 5,000 Words)
**Proposed Title**: *Adaptive Multimodal Fusion for Multi-Label ECG Diagnosis Using Deep Signal Learning and Clinical Metadata*

This template is structured to target a maximum length of 5,000 words upon expansion, making it suitable for a standard 4-to-6-page IEEE conference submission or letter. It contains 50–60% of the actual draft text, focusing on methodology, calibration results, and structural placeholders.

---

# Abstract

## Purpose
Summarize the study in a single paragraph (120 to 180 words).

## Starter Content
"Automated deep learning models for electrocardiogram (ECG) classification process waveforms in isolation, ignoring patient demographics. While multi-modal models integrate demographics, standard late-fusion methods rely on concatenation, which can propagate noise when ECG signals are degraded. We propose **Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF)** to address this vulnerability. PG-RAAF estimates an ECG reliability score ($R \in [0,1]$) from time-series feature maps to scale patient demographics before applying cross-attention. We evaluate the proposed model on the PTB-XL dataset, comparing it to 1D CNN, InceptionTime, Transformer ECG, and ResNet baselines. Under simulated noise (baseline drift, lead dropouts) and demographic chart mismatches, PG-RAAF outperforms concatenation methods. The model achieves a Test Macro F1-score of `[Insert F1]` and improves probability calibration, reducing Expected Calibration Error (ECE) to `[Insert ECE]` and Brier score to `[Insert Brier]`. The integration of Monte Carlo (MC) Dropout allows the model to flag low-confidence predictions, supporting safe clinical decision-making."

**Keywords**—Electrocardiography (ECG), explainable AI, multi-modal fusion, neural networks, reliability gating, uncertainty calibration.

---

# I. Introduction

## Purpose
Introduce the clinical context, state the limitations of existing methods, identify the research gap, and outline the contributions of the paper.

## Starter Content
"Electrocardiography (ECG) is a fundamental tool for assessing cardiac health, enabling clinicians to identify cardiovascular abnormalities. However, manual interpretation requires experienced cardiologists and is subject to inter-observer variability. Deep learning models have automated ECG analysis but process waveforms in isolation, ignoring demographic context (age, sex, height, weight) that clinicians use to interpret signals.

Multi-modal models have been proposed to integrate ECG signals and patient demographics. However, standard architectures rely on simple concatenation, which can propagate noise if the input signal is degraded (e.g., muscle artifacts or detached leads). This can lead to poor calibration and validation bias in clinical applications.

We propose the **Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF)** architecture to address these limitations. PG-RAAF estimates a reliability score ($R \in [0,1]$) directly from the ECG feature maps and scales demographic metadata projections accordingly before applying cross-attention. This ensures that the contribution of demographics is scaled down when the ECG signal is unreliable, maintaining stable performance and calibration."

## Expansion Notes
Students should expand the clinical background to approximately 300 words, focusing on cardiovascular statistics and 12-lead ECG guidelines.

## Required Figures
- `FIGURE REQUIRED`:
  - **Purpose**: Show standard concatenation vs. reliability-aware attention fusion gating.
  - **Caption**: *Fig. 1. Conceptual data flow showing standard concatenation vs. reliability-aware attention fusion.*
  - **Suggested filename**: `figure1_concept.png`

---

# II. Related Work

## Purpose
Contextualize the project within recent literature (2022–2025), discussing ECG classification, multi-modal fusion, and explainability.

## Starter Content
"Deep learning architectures, such as 1D CNNs, InceptionTime, ResNets, and ECG Transformers, have demonstrated high performance in automated ECG classification. However, these models ignore patient demographic context.

Multi-modal learning in healthcare aims to integrate heterogeneous data sources, such as raw waveforms and patient metadata. While joint representation learning has been explored, late-fusion concatenation remains the default, highlighting the need to investigate more adaptive fusion methods.

Explainable AI (XAI) is critical for clinical validation. Grad-CAM, Integrated Gradients, and SHAP value attributions have been applied to ECG networks to localize features on waveforms and demographics. However, most studies apply these methods to single modalities in isolation, missing cross-modal interactions."

## Expansion Notes
Students should expand this section to approximately 500 words, summarizing at least 8 recent papers (2020-2025) and constructing a literature comparison table.

## Required Tables
- `TABLE REQUIRED`:
  - **Caption**: *Table I. Comparison of Multi-Modal ECG Classification Approaches in Literature*
  - **Columns**: Paper, Dataset, Signal Network, Metadata Features, Fusion Method, Calibration, Explainability.

---

# III. Dataset & Preprocessing

## Purpose
Describe the dataset, patient cohorts, preprocessing pipeline, and splits used for model validation.

## Starter Content
"We evaluate our models on the PTB-XL dataset, which contains 21,837 clinical 12-lead ECG records from 18,885 patients. The dataset includes 5 diagnostic superclasses: Normal ECG (`NORM`), Myocardial Infarction (`MI`), ST/T Changes (`STTC`), Conduction Disturbance (`CD`), and Hypertrophy (`HYP`). Patient clinical demographics include age, sex, height, and weight.

We use a 10-fold patient-wise stratified split, where folds 1-8 are training, fold 9 is validation, and fold 10 is testing, using patient IDs to ensure no leakage across folds. ECG waveforms are sampled at 100 Hz and undergo zero-phase bandpass filtering from 0.5 Hz to 45 Hz using a 101-tap finite impulse response (FIR) filter to remove high-frequency noise and baseline drift. Demographic variables are imputed using the median values computed exclusively from training fold data, and numerical features (age, height, weight) are scaled to zero-mean and unit variance using standard scaling parameters fitted only on the training splits."

## Expansion Notes
Students should expand the dataset profile to approximately 400 words, discussing class imbalance and demographic cohort frequencies.

## Required Tables
- `TABLE REQUIRED`:
  - **Caption**: *Table II. Cohort Statistics and Class Frequencies*
  - **Columns**: Cohort Variable, Sample Count, Percentage (%), Class Label, Count.

---

# IV. Proposed Methodology

## Purpose
Detail the proposed Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF) architecture, including its mathematical formulation and data flow.

## Starter Content
"The proposed PG-RAAF architecture integrates 12-lead ECG waveforms and patient demographics using a reliability-aware attention fusion mechanism. The ECG waveform $X_{wave} \in \mathbb{R}^{12 \times 1000}$ is processed by a 1D ResNet branch to extract a signal embedding $h_{sig} \in \mathbb{R}^{64}$:
$$h_{sig} = f_{resnet}(X_{wave})$$
The patient demographic metadata $X_{meta} \in \mathbb{R}^4$ is processed by a 2-layer MLP to extract a tabular embedding $h_{meta} \in \mathbb{R}^{16}$:
$$h_{meta} = f_{mlp}(X_{meta})$$

To address signal noise and data entry errors, we introduce a physiology-guided reliability estimator. We compute a signal reliability score $R \in [0, 1]$ directly from the ECG feature maps:
$$R = \text{Sigmoid}(W_{r2} \cdot \text{ReLU}(W_{r1} h_{sig} + b_{r1}) + b_{r2})$$
The demographic embedding $h_{meta}$ is projected to the shared space $d_{model} = 64$ and scaled by the reliability score $R$:
$$\bar{h}_{meta} = \text{ReLU}(\text{BN}(W_m h_{meta} + b_m))$$
$$h_{meta}^{scaled} = R \cdot \bar{h}_{meta}$$

Using $h_{sig}$ as Query ($Q$), and $h_{meta}^{scaled}$ as Key ($K$) and Value ($V$), we apply cross-attention:
$$Q = W_q h_{sig}, \quad K = W_k h_{meta}^{scaled}, \quad V = W_v h_{meta}^{scaled}$$
$$h_{attn} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{model}}}\right) V$$
$$h_{fused} = W_{out} h_{attn} + h_{sig}$$
The fused embedding $h_{fused}$ is projected via a linear layer to predict diagnostic probabilities. The model is trained using Binary Cross Entropy (BCE) loss:
$$\mathcal{L} = - \frac{1}{C} \sum_{c=1}^C [y_c \log(p_c) + (1 - y_c) \log(1 - p_c)]$$"

## Expansion Notes
Students should expand this section to approximately 600 words, detailing the dimensions, hyperparameter weights, and loss function configurations.

## Required Figures
- `FIGURE REQUIRED`:
  - **Purpose**: Overall multi-modal pipeline diagram.
  - **Caption**: *Fig. 2. Detailed diagram of the multi-modal pipeline, showing the 1D-ResNet signal branch, metadata MLP, fusion module, and classifier head.*
  - **Suggested filename**: `figure_architecture.png`
- `FIGURE REQUIRED`:
  - **Purpose**: Fusion module block diagram.
  - **Caption**: *Fig. 3. Block diagram of the proposed Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF) mechanism, detailing how the reliability score scales demographic features before cross-attention.*
  - **Suggested filename**: `figure_fusion.png`

---

# V. Experimental Setup & Metrics

## Purpose
Document the experimental configuration, including hardware, libraries, and hyperparameters, to support reproducibility.

## Starter Content
"Our experimental setup is configured to support reproducibility. We optimize models using AdamW and a OneCycleLR learning rate scheduler. Training uses mixed precision (AMP) and early stopping based on validation loss. Random seeds are fixed globally (`seed=42`) across PyTorch, NumPy, and XGBoost to ensure reproducible execution. Model training is executed on an NVIDIA GPU using PyTorch (v2.0+) and standard data science libraries.

We evaluate model performance using Macro F1-score, Micro F1-score, Macro AUROC, and Macro AUPRC. To evaluate model calibration, we report Expected Calibration Error (ECE) and Brier scores. Expected Calibration Error (ECE) is formulated as:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where $B_m$ is the $m$-th confidence bin and $N$ is the number of samples. The multi-label Brier score is computed as:
$$\text{Brier} = \frac{1}{N \cdot C} \sum_{n=1}^N \sum_{c=1}^C (y_{n,c} - p_{n,c})^2$$
where $y_{n,c}$ is the binary ground-truth label and $p_{n,c}$ is the predicted probability for class $c$ on sample $n$."

## Required Tables
- `TABLE REQUIRED`:
  - **Caption**: *Table III. Model Training and Optimization Hyperparameters*
  - **Columns**: Hyperparameter Name, Configuration Value, Description.

## Required Figures
- `FIGURE REQUIRED`:
  - **Purpose**: Plot training and validation loss curves showing convergence.
  - **Caption**: *Fig. 4. Binary Cross-Entropy loss convergence curves on training and validation splits.*
  - **Suggested filename**: `figure1_loss_curves.png`
  - **Where it should appear**: Section V (Experimental Setup & Metrics) or Section VI (Results)
  - **How it should be generated**: Automatically generated and written to `paper_artifacts/figure1_loss_curves.png` (and PDF) during model training.

---

# VI. Results & Discussion

## Purpose
Present the classification performance, calibration metrics, baseline comparisons, ablation studies, and robustness sweeps.

## Starter Content
"We compare the performance of the proposed PG-RAAF model to single-modality baselines (XGBoost, 1D CNN, InceptionTime, Transformer ECG, ResNet ECG Only) and standard late-fusion methods. Table IV summarizes the classification performance, ECE, Brier scores, parameters, and inference latencies across models.

ROC and Precision-Recall curves are generated automatically. Fig. 5 plots the multi-class Receiver Operating Characteristic (ROC) curves, and Fig. 6 plots the multi-class Precision-Recall (PR) curves of the best PG-RAAF model on the test set. Fig. 7 displays the reliability calibration curves, showing predicted confidence against actual positive class frequencies. Fig. 8 shows the multi-class confusion matrices for the five diagnostic superclasses. Fig. 9 shows the relative demographic feature importance weights from the XGBoost baseline.

Table V summarizes the results of the ablation configurations, comparing F1, AUROC, ECE, Brier scores, and inference latencies. Table VI summarizes the results across the noise scenarios, demonstrating model robustness under simulated clinical noise and input errors. Table VII summarizes the Leave-Group-Out (LGO) generalization sweeps, simulating performance under gender-specific and age-specific clinical domain shifts."

## Required Figures
- `FIGURE REQUIRED`:
  - **Purpose**: Plot multi-class Receiver Operating Characteristic (ROC) curves.
  - **Caption**: *Fig. 5. Receiver Operating Characteristic (ROC) curves of the best PG-RAAF model on the test set.*
  - **Suggested filename**: `figure_roc_curves.png`
  - **Where it should appear**: Section VI (Results)
  - **How it should be generated**: Automatically generated and written to `paper_artifacts/figure_roc_curves.png` (and PDF) during model training.
- `FIGURE REQUIRED`:
  - **Purpose**: Plot multi-class Precision-Recall (PR) curves.
  - **Caption**: *Fig. 6. Precision-Recall (PR) curves of the best PG-RAAF model on the test set.*
  - **Suggested filename**: `figure_pr_curves.png`
  - **Where it should appear**: Section VI (Results)
  - **How it should be generated**: Automatically generated and written to `paper_artifacts/figure_pr_curves.png` (and PDF) during model training.
- `FIGURE REQUIRED`:
  - **Purpose**: Plot reliability calibration curves to show model calibration.
  - **Caption**: *Fig. 7. Reliability calibration curves comparing predicted confidence with actual positive sample ratios.*
  - **Suggested filename**: `figure_calibration_curves.png`
  - **Where it should appear**: Section VI (Results)
  - **How it should be generated**: Automatically generated and written to `paper_artifacts/figure_calibration_curves.png` (and PDF) during model training.
- `FIGURE REQUIRED`:
  - **Purpose**: Plot the multi-class Confusion Matrix.
  - **Caption**: *Fig. 8. Confusion matrices for the five diagnostic superclasses on the test set.*
  - **Suggested filename**: `figure_confusion_matrix.png`
  - **Where it should appear**: Section VI (Results)
  - **How it should be generated**: Automatically generated and written to `paper_artifacts/figure_confusion_matrix.png` (and PDF) during model training.
- `FIGURE REQUIRED`:
  - **Purpose**: Plot feature importance weights of patient demographic variables.
  - **Caption**: *Fig. 9. Demographic feature importance weights computed using Gini importance in the XGBoost baseline.*
  - **Suggested filename**: `figure_feature_importance.png`
  - **Where it should appear**: Section VI (Results)
  - **How it should be generated**: Automatically generated and written to `paper_artifacts/figure_feature_importance.png` (and PDF) during model training.

## Required Tables
- `TABLE REQUIRED`:
  - **Caption**: *Table IV. Baseline Comparison and Classifier Benchmarks*
  - **Columns**: Classifier Architecture, Macro F1, Micro F1, Macro AUROC, Macro AUPRC, ECE, Brier Score, Model Size (Params), Latency (ms).
- `TABLE REQUIRED`:
  - **Caption**: *Table V. Ablation Results for Different Fusion Configurations*
  - **Columns**: Ablation Configuration, Macro F1, Micro F1, Macro AUROC, Macro AUPRC, ECE, Brier Score.
- `TABLE REQUIRED`:
  - **Caption**: *Table VI. Model Performance Under Modality Degradation and Clinical Noise*
  - **Columns**: Noise Scenario, Macro F1, Micro F1, Macro AUROC, Macro AUPRC.
- `TABLE REQUIRED`:
  - **Caption**: *Table VII. Leave-Group-Out (LGO) Generalization Validation Sweeps Under Demographic Domain Shifts*
  - **Columns**: Experiment, Train Count, Test Count, Macro F1, Macro AUROC, ECE, Brier Score.

---

# VII. Explainability, Fairness & Statistical Validation

## Purpose
Evaluate model explainability, performance parity across demographic cohorts, and statistical significance of diagnostic improvements.

## Starter Content
"To provide explainability for clinical validation, we generate:
1. **Grad-CAM Temporal Attributions**: Displays visual overlays on the ECG waveforms showing which time segments contribute to predictions.
2. **Integrated Gradients Lead Attributions**: Highlights positive and negative contributions across the 12 leads.
3. **Shapley Demographic Attributions**: Visualizes how age, sex, weight, and height contribute to predictions.

Fig. 10 plots the Grad-CAM temporal attributions overlaid on the ECG Lead II signal. Fig. 11 shows the Integrated Gradients lead-wise attributions, and Fig. 12 shows the demographic feature attributions computed using exact Shapley values.

We evaluate model fairness across demographic cohorts (sex, age, and BMI) to ensure equitable diagnostic performance. Table VIII summarizes the results across sex, age, and BMI cohorts, reporting Macro F1, Micro F1, Macro AUROC, and Macro AUPRC for each cohort to evaluate potential bias.

To assess the statistical significance of performance improvements, we perform Wilcoxon signed-rank tests over sample-wise cross-entropy losses and McNemar tests over diagnostic classification outputs. Table IX summarizes the test statistics and p-values, verifying that the PG-RAAF network's diagnostic improvements are statistically significant ($p < 0.05$)."

## Required Figures
- `FIGURE REQUIRED`:
  - **Purpose**: Display Grad-CAM temporal attributions on Lead II.
  - **Caption**: *Fig. 10. Grad-CAM temporal attributions overlaid on the ECG Lead II signal.*
  - **Suggested filename**: `figure_attention_maps.png`
  - **Where it should appear**: Section VII (Explainability & Fairness)
  - **How it should be generated**: Automatically generated and written to `paper_artifacts/figure_attention_maps.png` (and PDF) during model training.
- `FIGURE REQUIRED`:
  - **Purpose**: Plot Integrated Gradients lead attributions.
  - **Caption**: *Fig. 11. Integrated Gradients lead-wise attributions across the 12 leads.*
  - **Suggested filename**: `figure_lead_contribution.png`
  - **Where it should appear**: Section VII (Explainability & Fairness)
  - **How it should be generated**: Automatically generated and written to `paper_artifacts/figure_lead_contribution.png` (and PDF) during model training.
- `FIGURE REQUIRED`:
  - **Purpose**: Plot demographic Shapley values.
  - **Caption**: *Fig. 12. Demographic feature attributions computed using exact Shapley values.*
  - **Suggested filename**: `figure_shap_plots.png`
  - **Where it should appear**: Section VII (Explainability & Fairness)
  - **How it should be generated**: Automatically generated and written to `paper_artifacts/figure_shap_plots.png` (and PDF) during model training.

## Required Tables
- `TABLE REQUIRED`:
  - **Caption**: *Table VIII. Performance Metrics across Demographic Cohorts*
  - **Columns**: Cohort Group, Sample Size, Macro F1, Micro F1, Macro AUROC, Macro AUPRC.
- `TABLE REQUIRED`:
  - **Caption**: *Table IX. Statistical Significance and Paired Diagnostics Significance*
  - **Columns**: Statistical Test, Test Statistic, p-value, Significant ($p < 0.05$).

---

# VIII. Limitations & Future Work

## Purpose
Acknowledge the boundaries of the current study and outline directions for future research.

## Starter Content
"This study has several limitations. First, the patient metadata consists of only four static variables: age, sex, weight, and height, lacking clinical context such as patient history or current medications. Second, the model is trained and evaluated using the PTB-XL dataset; external validation on separate datasets from different hospital systems is required to evaluate generalizability.

Future work will focus on implementing unified multi-modal Transformers (such as Perceiver or cross-attention encoders) to model interactions between raw signals and demographics, and integrating raw text from ECG clinical notes using pre-trained medical language models."

## Expansion Notes
Students should write approximately 400 words expanding on these limitations and future directions.

---

# IX. Conclusion

## Purpose
Summarize the main contributions and clinical impact of the study.

## Starter Content
"We proposed the Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF) architecture for multi-modal ECG diagnostics. By dynamically scaling demographic features based on ECG signal quality, the model maintains stable performance and calibration under simulated clinical noise and input errors. Our results demonstrate that PG-RAAF improves performance and calibration compared to standard late-fusion methods and single-modality baselines, providing a path toward reliable automated cardiac diagnostics."

---

# X. References

## Starter Content
`[1] P. Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," Scientific Data, 2020.`  
`[2] A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals," Circulation, 2000.`

## Expansion Notes
Students should include 25 to 35 references from recent biomedical AI and deep learning literature (2020-2025).
