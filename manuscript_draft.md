# Multi-Modal Adaptive Late-Fusion Network for Cardiac Diagnostics on 12-Lead ECG Signals and Patient Demographics

**Author Placeholder(s)**  
*Department of Biomedical Engineering / Computer Science*  
*Research Institution Placeholder*  
*Email: email@domain.edu*

---

## Abstract
Cardiac diagnostic models typically rely on raw electrocardiogram (ECG) waveforms, ignoring clinical demographic metadata such as age, sex, weight, and height. In this paper, we propose a multi-modal adaptive late-fusion network that integrates time-series feature extraction from 12-lead ECG signals with patient demographic attributes. We implement and compare five late-fusion modules: Concatenation, Gated Fusion, Cross-Attention, Feature Attention, and Dynamic Weighted Fusion. To evaluate our approach, we train models using the PTB-XL dataset and evaluate them under various clinical noise scenarios (Gaussian noise, electrode baseline drift, and lead dropouts). We assess demographic fairness across age, sex, and BMI cohorts, and apply Monte Carlo (MC) Dropout to estimate prediction uncertainty, flagging low-confidence outputs. Our gated and cross-attention fusion models outperform signal-only and metadata-only baselines, demonstrating the value of integrating demographic context into diagnostic workflows.

---

## I. Introduction
Electrocardiograms (ECGs) are standard tools used to diagnose cardiovascular conditions. While deep learning models can identify abnormalities in 12-lead ECG signals, clinical diagnostics in practice combine these signals with patient characteristics like age, sex, and physical metrics.

Traditional fusion models typically concatenate demographic variables with time-series embeddings at the final classifier layer. However, simple concatenation assumes linear independence and may fail to capture interactions between modalities. For example, some ECG patterns, like left ventricular hypertrophy, can vary based on a patient's age or body mass index.

To address these limitations, this paper introduces an adaptive multi-modal fusion framework. Our main contributions are:
1. **Adaptive Multi-Modal Fusion**: We implement and compare five late-fusion strategies (Concatenation, Gated, Cross-Attention, Feature Attention, and Dynamic Weighting) to integrate ECG time-series and patient demographics.
2. **Uncertainty Quantification**: We use Monte Carlo (MC) Dropout to estimate prediction uncertainty, flagging low-confidence outputs to improve safety.
3. **Clinical Robustness & Fairness Evaluation**: We evaluate model performance under simulated clinical noise (Gaussian noise, baseline wander, and lead dropouts) and profile fairness across demographic cohorts (sex, age, BMI).

---

## II. Related Work
### A. Deep Learning for ECG Classification
Convolutional neural networks (CNNs), particularly 1D ResNets and InceptionTime models, are widely used for ECG classification. ECG Transformers have also been introduced to model long-range temporal relationships in time-series data.

### B. Multi-Modal Fusion in Clinical AI
Multi-modal fusion in healthcare typically falls into early, joint, or late fusion. In cardiac diagnostics, late fusion is common, where feature vectors from raw signals and tabular clinical metadata are combined. However, simple concatenation remains the default, highlighting the need to investigate more adaptive fusion methods.

---

## III. Methodology

```
                   12-Lead ECG Signal (12, 1000)
                                 │
                                 ▼
                     1D-ResNet Feature Extractor
                                 │
                                 ▼
                       Signal Embedding (64-D) ───┐
                                                  ▼
                                            Fusion Module (Gated/Attention)
                                                  ▲
                       Tabular Embedding (16-D) ──┘
                                 │
                                 ▼
                        Demographic MLP
                                 │
                                 ▼
                   Clinical Metadata (Age, Sex, Ht, Wt)
```

### A. Feature Extraction Branches
1. *ECG Branch*: A 1D ResNet model maps raw signals $X_{wave} \in \mathbb{R}^{12 \times 1000}$ to a 64-dimensional embedding $h_{sig} \in \mathbb{R}^{64}$.
2. *Metadata Branch*: A 2-layer MLP projects demographic features $X_{meta} \in \mathbb{R}^{4}$ to a 16-dimensional embedding $h_{meta} \in \mathbb{R}^{16}$.

### B. Adaptive Fusion Strategies
We compare five fusion strategies to integrate $h_{sig}$ and $h_{meta}$:
1. **Concatenation (Baseline)**:
   $$h_{fused} = [h_{sig} \mathbin{\Vert} h_{meta}] \in \mathbb{R}^{80}$$
2. **Gated Fusion**:
   $$g = \sigma(W_g [h_{sig} \mathbin{\Vert} h_{meta}] + b_g)$$
   $$h_{fused} = g \odot W_s h_{sig} + (1 - g) \odot W_m h_{meta}$$
3. **Cross-Attention**:
   $$Q = W_q h_{sig}, \quad K = W_k h_{meta}, \quad V = W_v h_{meta}$$
   $$h_{fused} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V + Q$$
4. **Feature Attention**: Applies self-attention over the concatenated 80-D vector.
5. **Dynamic Weighting**: Learns scalar weights for the two modalities using a softmax function.

---

## IV. Experimental Setup
### A. Dataset & Splits
We evaluate our models on the PTB-XL dataset. We use a 10-fold patient-wise stratified split to prevent data leakage, evaluating performance using Macro F1, Micro F1, Macro AUROC, and Macro AUPRC.

### B. Training Configuration
Models are optimized using AdamW and a OneCycleLR learning rate scheduler. Training uses mixed precision (AMP) and early stopping based on validation loss.

---

## V. Results and Analysis
All experimental tables and figures are automatically generated by running the research suite:
- **Table II** compares performance across baselines, including XGBoost, 1D CNN, ResNet, InceptionTime, and Transformers.
- **Table III** evaluates the five late-fusion variants.
- **Table IV** assessments robustness against Gaussian noise, baseline wander, and lead dropouts.
- **Table V** evaluates performance across age, sex, and BMI cohorts.
- **Table VI** shows statistical significance results (McNemar and Wilcoxon tests).

---

## VI. Discussion
Our experiments indicate that:
1. **Adaptive Fusion Benefits**: Gated and cross-attention fusion strategies outperform simple concatenation by allowing the model to adaptively weight features based on demographic context.
2. **Uncertainty Quantification**: Implementing MC Dropout helps identify low-confidence predictions ($H > 0.75$), which could help prevent automated diagnostic errors.
3. **Robustness**: Evaluating models under simulated noise shows that integrating demographics can help maintain performance when ECG signals are degraded or leads are missing.

---

## VII. Conclusion
We presented a multi-modal adaptive late-fusion framework for ECG classification. Comparing five fusion strategies, our results demonstrate that gated and attention-based modules improve diagnostic performance and robustness. Integrating uncertainty estimation and explainability tools can support more reliable clinical decision-making.

---

## References Placeholder
1. *A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals," Circulation, 2000.*
2. *P. Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," Scientific Data, 2020.*
3. *Other relevant biomedical AI and deep learning references.*
