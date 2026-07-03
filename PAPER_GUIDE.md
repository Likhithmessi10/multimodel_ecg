# IEEE Manuscript Preparation Guide & Writing Blueprint
**Target Venue**: IEEE Transactions on Biomedical Engineering / IEEE Transactions on Pattern Analysis and Machine Intelligence  
**Model Focus**: Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF) on PTB-XL

This document serves as a complete writing manual for co-authors to draft an IEEE-style paper based on this repository. Follow the sections, structural guidelines, mathematical formulations, and reviewer expectations detailed below.

---

# 1. Title

## Purpose
The title must clearly communicate the core contribution: multi-modal classification using 12-lead ECG signals and patient demographics with a reliability-aware attention fusion mechanism. It should be concise, professional, and contain active terms.

## How to Create a Good Title
- **Incorporate key modalities**: Mention both "12-Lead Electrocardiogram" (or "ECG time-series") and "Clinical Metadata" (or "Patient Demographics").
- **Highlight the core novelty**: Reference "Reliability-Aware Attention" or "Adaptive Fusion".
- **Avoid jargon**: Do not use vague terms like "smart," "novel," or "next-generation."
- **Expected Length**: 10 to 14 words.

## Examples
- **Weak Title**: *Using Deep Learning to Classify ECG Signals with Patient Data* (Too generic, lacks detail on the fusion method and clinical context).
- **Weak Title**: *A Smart and Novel Framework for Multi-Modal ECG Diagnostics* (Uses buzzwords, lacks technical specificity).
- **Strong Title**: *Multi-Modal Cardiac Diagnostics via Physiology-Guided Reliability-Aware Attention Fusion of 12-Lead ECGs and Patient Demographics* (Specific, highlights the core method, and lists the inputs).
- **Strong Title**: *Uncertainty-Aware ECG Classification with Metadata-Conditioned Reliability Gating and Cross-Attention* (Highlights uncertainty and the specific gating mechanism).

## Section Checklist
- [ ] Contains no buzzwords ("smart", "novel", "SOTA").
- [ ] Explicitly mentions the inputs: 12-lead ECG waveforms and patient demographics.
- [ ] References the reliability-aware or adaptive attention fusion mechanism.
- [ ] Under 15 words.

---

# 2. Abstract

## Purpose
The abstract is a single-paragraph summary of the entire paper (typically 150 to 250 words) that describes the background, problem statement, research gap, methodology, key findings, and clinical impact.

## Suggested Structure & Paragraph Flow
1. **Background (1-2 sentences)**: Introduce the clinical importance of automated 12-lead ECG diagnostics.
2. **Problem Statement & Gap (2 sentences)**: Explain that while deep learning models process raw signals, they often ignore patient demographics. Note that existing multi-modal models rely on simple concatenation, which can propagate noise when signals are degraded or records are incomplete.
3. **Proposed Method (2 sentences)**: Introduce the Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF) model, which dynamically scales demographic features based on ECG signal quality.
4. **Experimental Setup & Key Results (2 sentences)**: Summarize validation on the PTB-XL dataset under noise (baseline wander, lead dropout) and demographic domain shifts. Mention the baseline models compared.
5. **Clinical Impact (1 sentence)**: Explain how reliability gating and uncertainty calibration (using ECE and Brier score) support clinical decision-making.

## Keywords
Provide 4 to 6 index terms (in alphabetical order) selected from the IEEE Taxonomy:
*Calibration, electrocardiography (ECG), explainable AI, multi-modal fusion, neural networks, uncertainty estimation.*

## Section Checklist
- [ ] Written as a single, continuous paragraph (no indents or line breaks).
- [ ] Between 150 and 250 words.
- [ ] Clearly states the research gap (concatenation vulnerability to noise).
- [ ] Summarizes the proposed PG-RAAF method.
- [ ] Highlights calibration (ECE) and robustness findings.
- [ ] Ends with a statement on clinical significance.

---

# 3. Introduction

## Purpose
The introduction outlines the research context, details the limitations of existing methods, identifies the research gap, and states the paper's contributions.

## Subsections & Paragraph Flow

### A. Clinical Background & Importance
- **What to Write**: Discuss cardiovascular diseases (CVDs) as a leading cause of mortality globally. Explain the role of the 12-lead ECG as a non-invasive diagnostic tool.
- **Citations Required**:
  - `CITATION REQUIRED`: World Health Organization (WHO) cardiovascular statistics (2020-2024).
  - `CITATION REQUIRED`: Clinical guidelines on standard 12-lead ECG interpretation (e.g., ACC/AHA guidelines).

### B. Problem Statement & Traditional Modality Limits
- **What to Write**: Detail how clinicians interpret ECGs in the context of patient demographics (age, sex, height, weight). Note that signal-only deep learning models miss this demographic context.
- **Citations Required**:
  - `CITATION REQUIRED`: Studies showing how physiological variations (e.g., chest size in obesity, age-related changes) affect ECG amplitudes.

### C. Existing Limitations & the Multi-Modal Fusion Gap
- **What to Write**: Critique existing late-fusion approaches. Explain that simple concatenation (`torch.cat`) assumes linear independence and can propagate noise if the input signal is degraded (e.g., baseline drift, detached leads) or metadata is incomplete.
- **Figures Required**:
  - `FIGURE REQUIRED`:
    - **Purpose**: Concept comparison showing standard concatenation vs. reliability-aware attention fusion.
    - **Caption**: *Fig. 1. Conceptual overview showing how standard concatenation propagates noise vs. our reliability-aware attention fusion gating.*
    - **Suggested filename**: `figure1_concept.png`
    - **How to generate**: Create a flowchart in PowerPoint, Draw.io, or TikZ showing the comparative data flow.
    - **Expected size**: Single-column width (approx. 3.5 inches).
    - **Expected quality**: Vector format (PDF) or 300 DPI PNG.

### D. Proposed Solution & Core Contributions
- **What to Write**: Introduce the PG-RAAF model. Explain how it estimates signal reliability ($R$) to scale patient metadata before cross-attention. Bullet the contributions:
  1. Design of the PG-RAAF block for noise-resilient multi-modal ECG diagnostics.
  2. Integration of Expected Calibration Error (ECE) and Brier score metrics to evaluate model calibration.
  3. Evaluation of robustness under clinical noise, missing leads, and demographic chart mismatches.
  4. Global explainability analysis using Grad-CAM, Integrated Gradients, and exact demographic Shapley values.

### E. Paper Organization
- **What to Write**: Provide a brief outline of the remaining sections of the paper:
  - *Section II discusses related work. Section III outlines dataset preprocessing. Section IV details the methodology. Section V describes the experimental setup, followed by results and discussion in Sections VI and VII.*

## Reviewer Expectations
- Reviewers look for a clear statement of the research gap. Do not just claim to "improve accuracy."
- Ensure that every contribution listed in the introduction is supported by experiments in the results section.

---

# 4. Related Work

## Purpose
The related work section contextualizes the project within recent literature (2022–2025), discussing ECG classification, multi-modal fusion, explainable AI, and clinical decision support.

## Subsections
1. **ECG Classification & Deep Time-Series Models**: Discuss 1D CNNs, InceptionTime, ResNets, and ECG Transformers.
2. **Multi-Modal Learning in Healthcare**: Review fusion methods (early, joint, late) and point out that late fusion concatenation is the standard.
3. **Explainable AI (XAI) in Electrocardiography**: Discuss Grad-CAM, Integrated Gradients, and SHAP. Highlight the limitation that most studies apply these only to single modalities.
4. **Clinical Uncertainty and Calibration**: Discuss the importance of ECE and calibration in safety-critical medical applications.

## Required Literature Comparison Table
Include a comparison table summarizing recent publications to highlight the gap:

- `TABLE REQUIRED`:
  - **Caption**: *Table I. Comparison of Multi-Modal ECG Classification Approaches in Literature*
  - **Columns**: Paper, Dataset, Signal Network, Metadata Features, Fusion Method, Calibration, Explainability, Robustness Analysis.
  - **Rows**: List 4-5 representative papers (e.g., Paper A, B, C, D from Phase 2) alongside the proposed PG-RAAF model to demonstrate the methodology differences.
  - **Purpose**: To show that the proposed framework integrates calibration, explainability, and robustness sweeps, which are missing in prior works.

---

# 5. Dataset

## Purpose
This section describes the dataset, patient cohorts, label structures, preprocessing pipeline, and splits used for model validation.

## Content & Details
- **PTB-XL Profile**: A publicly available dataset containing 21,837 clinical 12-lead ECG records from 18,885 patients.
- **Labels**: 5 diagnostic superclasses: Normal ECG (`NORM`), Myocardial Infarction (`MI`), ST/T Changes (`STTC`), Conduction Disturbance (`CD`), and Hypertrophy (`HYP`).
- **Demographics**: 4 variables: Age, Sex (Male/Female), Height, and Weight.
- **Splits**: Standard 10-fold split where folds 1-8 are training, fold 9 is validation, and fold 10 is testing, using patient IDs to ensure no leakage across folds.
- **Citations Required**:
  - `CITATION REQUIRED`: *PTB-XL dataset paper (Wagner et al., Scientific Data, 2020).*
  - `CITATION REQUIRED`: *PhysioNet platform reference (Goldberger et al., Circulation, 2000).*

## Preprocessing Pipeline
1. **ECG Filtering**: 101-tap finite impulse response (FIR) zero-phase bandpass filter (0.5 Hz - 45 Hz) to remove muscle artifacts and baseline drift.
2. **Imputation**: Demographics are imputed using the median values computed *exclusively* from training fold data.
3. **Scaling**: Numerical features (age, height, weight) are scaled to zero-mean and unit variance using standard scaling parameters fitted *only* on the training splits.

## Figures Required
- `FIGURE REQUIRED`:
  - **Purpose**: Show class distribution across the five diagnostic superclasses.
  - **Caption**: *Fig. 2. Class frequency distribution of the PTB-XL diagnostic superclasses.*
  - **Suggested filename**: `figure_dataset_distribution.png`
  - **How to generate**: Plot a horizontal bar chart of the class counts in `matplotlib`.
  - **Expected size**: Single-column width.

## Tables Required
- `TABLE REQUIRED`:
  - **Caption**: *Table II. Cohort Statistics and Class Frequencies*
  - **Columns**: Cohort Variable, Sample Count, Percentage (%), Class Label, Count.
  - **Rows**: Demographic variables (Sex: Male/Female; Age: <65/$\ge$65) and diagnostic superclasses.
  - **Purpose**: To show the characteristics and class balance of the dataset.

---

# 6. Proposed Methodology

## Purpose
This section details the proposed Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF) architecture, including its mathematical formulation and data flow.

## Mathematical Formulation

### A. Modality Branches
Let the input ECG waveform be $X_{wave} \in \mathbb{R}^{12 \times 1000}$ and the input demographics be $X_{meta} \in \mathbb{R}^4$.
- The 1D-ResNet maps $X_{wave}$ to a signal embedding:
  $$h_{sig} = f_{resnet}(X_{wave}) \in \mathbb{R}^{64}$$
- The MLP maps $X_{meta}$ to a tabular embedding:
  $$h_{meta} = f_{mlp}(X_{meta}) \in \mathbb{R}^{16}$$

- `EQUATION REQUIRED`:
  - **Equation purpose**: Define the 1D Residual connection inside the ECG signal branch:
    $$y = \text{MaxPool1D}(\text{ReLU}(\text{BN}(\text{Conv1D}(x))) + W_s x)$$
  - **Variables**: $x$ (input feature map), $y$ (output feature map), $W_s$ (shortcut Conv1D projection projection parameters).

### B. Physiology-Guided Reliability Estimation
We compute a signal reliability score $R \in [0, 1]$ directly from the ECG feature maps:
$$R = \sigma(W_r h_{sig} + b_r)$$
- `EQUATION REQUIRED`:
  - **Equation purpose**: Define the learnable reliability score $R$:
    $$R = \text{Sigmoid}(W_{r2} \cdot \text{ReLU}(W_{r1} h_{sig} + b_{r1}) + b_{r2})$$
  - **Variables**: $W_{r1}, W_{r2}$ (projection weights), $b_{r1}, b_{r2}$ (biases), $h_{sig}$ (64-D signal embedding).

### C. Reliability-Aware Demographic Scaling
The clinical metadata embedding $h_{meta}$ is projected to the shared space $d_{model} = 64$ and scaled by the reliability score $R$:
$$\bar{h}_{meta} = \text{ReLU}(\text{BN}(W_m h_{meta} + b_m)) \in \mathbb{R}^{64}$$
$$h_{meta}^{scaled} = R \cdot \bar{h}_{meta} \in \mathbb{R}^{64}$$
If the ECG signal is noisy or degraded, $R \to 0$, scaling down the demographic features.

### D. Cross-Attention Fusion
Using $h_{sig}$ as Query ($Q$), and $h_{meta}^{scaled}$ as Key ($K$) and Value ($V$):
$$Q = W_q h_{sig}, \quad K = W_k h_{meta}^{scaled}, \quad V = W_v h_{meta}^{scaled}$$
- `EQUATION REQUIRED`:
  - **Equation purpose**: Define the scaled dot-product cross-attention:
    $$h_{attn} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{model}}}\right) V$$
    $$h_{fused} = W_{out} h_{attn} + h_{sig}$$
  - **Variables**: $d_{model} = 64$ (scaling factor), $W_{out}$ (projection weights).

### E. Loss Function
Multi-label training uses Binary Cross Entropy (BCE) loss:
$$\mathcal{L} = - \frac{1}{C} \sum_{c=1}^C [y_c \log(p_c) + (1 - y_c) \log(1 - p_c)]$$
where $y_c \in \{0, 1\}$ is the ground truth label and $p_c \in [0, 1]$ is the predicted probability for class $c$.

## Pseudocode
Include pseudocode detailing the PG-RAAF forward pass:

```python
# Pseudocode for PG-RAAF Fusion Block Forward Pass
def pg_raaf_forward(x_wave, x_meta):
    # Step 1: Extract embeddings
    h_sig = signal_branch(x_wave)  # Shape: (B, 64)
    h_meta = meta_branch(x_meta)   # Shape: (B, 16)
    
    # Step 2: Estimate signal reliability R
    R = sigmoid(linear2(relu(linear1(h_sig))))  # Shape: (B, 1)
    
    # Step 3: Project and scale demographic features
    h_meta_proj = relu(bn(linear_proj(h_meta)))  # Shape: (B, 64)
    h_meta_scaled = R * h_meta_proj              # Shape: (B, 64)
    
    # Step 4: Compute Query, Key, and Value
    Q = W_q(h_sig).unsqueeze(1)            # Shape: (B, 1, 64)
    K = W_k(h_meta_scaled).unsqueeze(1)    # Shape: (B, 1, 64)
    V = W_v(h_meta_scaled).unsqueeze(1)    # Shape: (B, 1, 64)
    
    # Step 5: Scaled dot-product attention
    scores = softmax((Q @ K.transpose(-2, -1)) / sqrt(64))  # Shape: (B, 1, 1)
    attn_out = (scores @ V).squeeze(1)                      # Shape: (B, 64)
    
    # Step 6: Final projection with residual connection
    h_fused = W_out(attn_out) + h_sig  # Shape: (B, 64)
    return classifier(h_fused)         # Shape: (B, 5)
```

## Section Checklist
- [ ] Includes the block diagrams of the signal and metadata branches.
- [ ] Contains the equations for reliability score computation.
- [ ] Formulates the cross-attention equations with dimensions.
- [ ] Shows the pseudocode of the PG-RAAF forward pass.
- [ ] Specifies the Loss Function mathematically.

---

## 7. Model Architecture Diagrams

## Purpose
This section provides visualizations of the overall pipeline and the PG-RAAF module.

## Required Figures
- `FIGURE REQUIRED`:
  - **Purpose**: Overall multi-modal pipeline diagram.
  - **Caption**: *Fig. 3. Detailed diagram of the multi-modal pipeline, showing the 1D-ResNet signal branch, metadata MLP, fusion module, and classifier head.*
  - **Suggested filename**: `figure_architecture.png`
  - **How to generate**: Exported during training. The file is saved at `paper_artifacts/figure_architecture.png` (and PDF).
  - **Expected size**: Double-column width.
  - **Expected quality**: 300 DPI, vector graphics text.

- `FIGURE REQUIRED`:
  - **Purpose**: Fusion module block diagram.
  - **Caption**: *Fig. 4. Block diagram of the proposed Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF) mechanism, detailing how the reliability score scales demographic features before cross-attention.*
  - **Suggested filename**: `figure_fusion.png`
  - **How to generate**: Exported during training. The file is saved at `paper_artifacts/figure_fusion.png` (and PDF).
  - **Expected size**: Single-column width.

---

# 8. Experimental Setup

## Purpose
This section documents the experimental configuration, including hardware, libraries, and hyperparameters, to support reproducibility.

## Hardware & Software Specifications
- **Hardware**: CPU/GPU specifications used for training (e.g., NVIDIA RTX series GPU or CPU fallback).
- **Libraries**: PyTorch (v2.0+), WFDB, Scikit-Learn, Scipy, XGBoost, and Matplotlib.
- **Reproducibility**: Random seed fixed globally (`seed=42`) across PyTorch, NumPy, and XGBoost. Hyperparameters are managed using `config.json`.

## Required Configuration Table
- `TABLE REQUIRED`:
  - **Caption**: *Table III. Model Training and Optimization Hyperparameters*
  - **Columns**: Hyperparameter Name, Configuration Value, Description.
  - **Rows**: Seed, Epochs, Batch Size, Optimizer, Initial Learning Rate, Weight Decay, MC Dropout rate, MC Steps, Bootstrap iterations.
  - **Purpose**: To document the training settings.
  - **Repository files contributing**: `paper_artifacts/table_hyperparameters.csv`.

---

# 9. Results

## Purpose
This section presents the classification performance, calibration metrics, and baseline comparisons.

## Required Figures
- `FIGURE REQUIRED`:
  - **Purpose**: Plot multi-class Receiver Operating Characteristic (ROC) curves.
  - **Caption**: *Fig. 5. Receiver Operating Characteristic (ROC) curves of the best PG-RAAF model on the test set.*
  - **Suggested filename**: `figure_roc_curves.png`
  - **How to generate**: Exported during training. The file is saved at `paper_artifacts/figure_roc_curves.png` (and PDF).
  - **Expected size**: Single-column width.

- `FIGURE REQUIRED`:
  - **Purpose**: Plot multi-class Precision-Recall (PR) curves.
  - **Caption**: *Fig. 6. Precision-Recall (PR) curves of the best PG-RAAF model on the test set.*
  - **Suggested filename**: `figure_pr_curves.png`
  - **How to generate**: Exported during training. The file is saved at `paper_artifacts/figure_pr_curves.png` (and PDF).
  - **Expected size**: Single-column width.

- `FIGURE REQUIRED`:
  - **Purpose**: Plot the multi-class Confusion Matrix.
  - **Caption**: *Fig. 7. Confusion matrices for the five diagnostic superclasses on the test set.*
  - **Suggested filename**: `figure_confusion_matrix.png`
  - **How to generate**: Exported during training. The file is saved at `paper_artifacts/figure_confusion_matrix.png` (and PDF).
  - **Expected size**: Double-column width.

## Required Tables
- `TABLE REQUIRED`:
  - **Caption**: *Table IV. Baseline Comparison and Classifier Benchmarks*
  - **Columns**: Classifier Architecture, Macro F1, Micro F1, Macro AUROC, Macro AUPRC, ECE, Brier Score, Model Size (Params), Latency (ms).
  - **Rows**: XGBoost, 1D CNN, InceptionTime, Transformer ECG, ResNet ECG Only, Proposed PG-RAAF.
  - **Purpose**: To show how the proposed PG-RAAF model compares to standard baseline models.
  - **Repository files contributing**: `paper_artifacts/table_model_comparisons.csv`.

---

# 10. Ablation Study

## Purpose
The ablation study evaluates the impact of each module by systematically removing components of the proposed architecture.

## Experiments to Perform
1. **ECG only**: Signal-only model (ResNet branch).
2. **Metadata only**: Tabular-only model (XGBoost baseline).
3. **Late Fusion (Concat)**: Fuses features using concatenation.
4. **Late Fusion (Gated)**: Fuses features using gated weights.
5. **Cross-Attention**: Fuses features using cross-attention without reliability scaling.
6. **Proposed PG-RAAF**: The full model with reliability-scaled attention.

## Tables Required
- `TABLE REQUIRED`:
  - **Caption**: *Table V. Ablation Results for Different Fusion Configurations*
  - **Columns**: Ablation Configuration, Macro F1, Micro F1, Macro AUROC, Macro AUPRC, ECE, Brier Score, Inference Latency (ms).
  - **Rows**: The 6 configurations listed above.
  - **Purpose**: To evaluate the performance impact of each component of the proposed model.
  - **Repository files contributing**: `paper_artifacts/table_ablation_results.csv`.

---

# 11. Robustness Analysis

## Purpose
This section evaluates model resilience under simulated clinical noise, missing leads, and demographic chart mismatches.

## Experiments to Perform
1. **Clean Baseline**: Evaluates performance on clean inputs.
2. **Missing Demographic Features**: Removing individual demographics (Age, Sex, Height, Weight) or all demographics.
3. **Incorrect Demographics (Chart Swaps)**: Swapping demographics between patients to simulate data entry errors.
4. **Gaussian Noise**: Adding noise to the ECG signal at different levels ($\sigma \in \{0.05, 0.15, 0.30\}$).
5. **Baseline Wander**: Adding sinusoidal noise (0.15 Hz) at different amplitudes ($A \in \{0.2, 0.4, 0.6\}$ mV).
6. **Lead Dropout**: Randomly dropping leads at different ratios ($r \in \{0.17, 0.33, 0.50\}$).

## Tables Required
- `TABLE REQUIRED`:
  - **Caption**: *Table VI. Model Performance Under Modality Degradation and Clinical Noise*
  - **Columns**: Noise Scenario, Macro F1, Micro F1, Macro AUROC, Macro AUPRC.
  - **Rows**: The 13 noise scenarios listed above.
  - **Purpose**: To evaluate model robustness under clinical noise and input errors.
  - **Repository files contributing**: `paper_artifacts/table_robustness_sweeps.csv`.

---

# 12. Explainability

## Purpose
This section evaluates model explainability, showing how the model localizes features on the ECG waveform and demographics.

## Attributions to Generate
1. **Grad-CAM Temporal Attributions**: Displays visual overlays on the ECG waveforms showing which time segments contribute to predictions.
2. **Integrated Gradients Lead Attributions**: Highlights positive and negative contributions across the 12 leads.
3. **Shapley Demographic Attributions**: Visualizes how age, sex, weight, and height contribute to predictions.

## Figures Required
- `FIGURE REQUIRED`:
  - **Purpose**: Display Grad-CAM temporal attributions on Lead II.
  - **Caption**: *Fig. 8. Grad-CAM temporal attributions overlaid on the ECG Lead II signal.*
  - **Suggested filename**: `figure_attention_maps.png`
  - **How to generate**: Exported during training. The file is saved at `paper_artifacts/figure_attention_maps.png` (and PDF).
  - **Expected size**: Single-column width.

- `FIGURE REQUIRED`:
  - **Purpose**: Plot Integrated Gradients lead attributions.
  - **Caption**: *Fig. 9. Integrated Gradients lead-wise attributions across the 12 leads.*
  - **Suggested filename**: `figure_lead_contribution.png`
  - **How to generate**: Exported during training. The file is saved at `paper_artifacts/figure_lead_contribution.png` (and PDF).
  - **Expected size**: Double-column width.

- `FIGURE REQUIRED`:
  - **Purpose**: Plot demographic Shapley values.
  - **Caption**: *Fig. 10. Demographic feature attributions computed using exact Shapley values.*
  - **Suggested filename**: `figure_shap_plots.png`
  - **How to generate**: Exported during training. The file is saved at `paper_artifacts/figure_shap_plots.png` (and PDF).
  - **Expected size**: Single-column width.

---

# 13. Uncertainty Analysis & Calibration

## Purpose
This section evaluates prediction calibration and uncertainty using Expected Calibration Error (ECE) and Brier scores.

## Content & Reliability Diagrams
Explain how well-calibrated probabilities are computed. Plot reliability diagrams to compare predicted confidence with accuracy.

## Figures Required
- `FIGURE REQUIRED`:
  - **Purpose**: Reliability diagram showing model calibration.
  - **Caption**: *Fig. 11. Reliability diagram showing predicted confidence vs. actual accuracy on the test set.*
  - **Suggested filename**: `figure_calibration_curves.png`
  - **How to generate**: Exported during training. The file is saved at `paper_artifacts/figure_calibration_curves.png` (and PDF).
  - **Expected size**: Single-column width.

---

# 14. Demographic Fairness Analysis

## Purpose
This section evaluates performance parity across demographic cohorts to check for potential bias.

## Cohorts to Compare
- **Sex**: Male vs. Female.
- **Age**: Young (age < 65) vs. Elderly (age $\ge$ 65).
- **BMI**: Normal (BMI < 25) vs. Overweight (BMI $\ge$ 25).

## Tables Required
- `TABLE REQUIRED`:
  - **Caption**: *Table VII. Performance Metrics across Demographic Cohorts*
  - **Columns**: Cohort Group, Sample Size, Macro F1, Micro F1, Macro AUROC, Macro AUPRC.
  - **Rows**: The 6 cohorts listed above.
  - **Purpose**: To evaluate model fairness across demographic groups.
  - **Repository files contributing**: `paper_artifacts/table_fairness_cohorts.csv`.

---

# 15. Statistical Validation

## Purpose
This section presents the statistical tests used to validate the model's performance improvements.

## Statistical Tests
- **McNemar Chi-Squared Test**: Computes the significance of accuracy differences between models.
- **Wilcoxon Signed-Rank Test**: A paired test evaluated over sample-wise BCE losses.
- **Bootstrap Confidence Intervals**: Evaluates 95% confidence intervals for F1 and AUROC over 200 iterations.

## Tables Required
- `TABLE REQUIRED`:
  - **Caption**: *Table VIII. Statistical Significance and Confidence Intervals*
  - **Columns**: Statistical test, Test Statistic, p-value, Significant ($p < 0.05$).
  - **Rows**: Proposed vs. XGBoost comparisons and Bootstrap intervals.
  - **Purpose**: To show the statistical significance of the model's improvements.
  - **Repository files contributing**: `paper_artifacts/table_statistical_tests.csv`.

---

# 16. Computational Efficiency Analysis

## Purpose
This section evaluates the model's computational footprint and suitability for clinical deployment.

## Content & Details
Compare parameter counts, model file size on disk, and CPU/GPU inference latency (ms per sample) to evaluate the computational cost of the proposed PG-RAAF model.

## Tables Required
- `TABLE REQUIRED**:
  - **Caption**: *Table IX. Model Parameter Counts and Inference Latency*
  - **Columns**: Model Architecture, Parameters, Latency on CPU (ms).
  - **Rows**: XGBoost, 1D CNN, InceptionTime, Transformer ECG, ResNet ECG Only, Proposed PG-RAAF.
  - **Purpose**: To compare the computational efficiency of the models.
  - **Repository files contributing**: `paper_artifacts/table_model_comparisons.csv`.

---

# 17. Discussion

## Suggested Paragraph Structure
1. **Interpretation of Findings**: Discuss how the proposed PG-RAAF model improves performance and calibration compared to the baseline models.
2. **Robustness**: Explain why the reliability estimator helps maintain stability under ECG noise and missing metadata.
3. **Explainability**: Discuss how combining Grad-CAM, Integrated Gradients, and Shapley values supports clinical validation.
4. **Clinical Deployment**: Address the practical steps for integrating the model into electronic health records (EHRs).

---

# 18. Limitations

## Content Guidelines
- **Demographic Features**: The patient metadata consists of only four static variables: age, sex, weight, and height. It lacks clinical context such as patient history, comorbidities, lab values, or current medications.
- **Single-Center Focus**: The model is trained and evaluated using the PTB-XL dataset. External validation on separate datasets from different hospital systems is required to evaluate generalizability.
- **Time-Series Length**: The model requires input waveforms to be exactly 10 seconds long (1000 samples).

---

# 19. Future Work

## Content Guidelines
- **Multimodal Transformers**: Implementing unified multi-modal Transformers (e.g., Perceiver or cross-attention encoders) to model interactions between raw signals and demographics.
- **Clinical Narrative Integration**: Integrating raw text from ECG clinical notes and reports using pre-trained medical language models (e.g., ClinicalBERT).
- **Quantization**: Quantizing the model (FP16 or INT8 precision) to evaluate latency reductions on edge devices.

---

# 20. Conclusion

## Suggested Paragraph Structure
- **Main Contribution**: Restate the PG-RAAF architecture and its novelty.
- **Impact**: Discuss how the model's calibration and robustness help support safe clinical decision-making.
- **Closing**: Outline future directions.

---

# 21. References

## Style Guidelines
- Follow standard IEEE citation format: `[1] Author, "Title," Journal, vol., no., pp., year.`
- Include 30 to 45 references, focusing on seminal deep learning papers (e.g., ResNet, InceptionTime) and recent multi-modal medical AI studies (2020-2025).

---

# 22. Appendix

## Suggested Content
- **Appendix A: Detailed Model Layer Dimensions**: Table mapping the channels, kernel sizes, and activation layouts of the 1D-ResNet branch.
- **Appendix B: Hyperparameter Tuning Sweep Details**: Accuracy variations across different cross-attention model setups during hyperparameter tuning.
- **Appendix C: Code Snippets**: Python class of the PG-RAAF module.
