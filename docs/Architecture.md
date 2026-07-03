# System Architecture & Multi-Modal Topology

This document details the multi-modal neural network topology designed for 12-lead ECG classification combined with patient demographic features.

---

## 1. Modality Branches

Our framework ingests two heterogeneous data modalities: 1D time-series signals and 1D static patient clinical metadata.

```
                    ┌───────────────────────────┐
                    │ Raw 12-Lead ECG waveform  │
                    │      (12, 1000)           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │      1D-ResNet CNN        │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Signal Embedding (64-D)   │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
  ┌───────────────────────────────┴──────────────────────────────┐
  │                        Fusion Module                         │
  │     (Concatenation, Gating, Cross-Attention, Feature-Attn)   │
  └───────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │  Classifier Decision Head │
                    │    Sigmoid Outputs (5)    │
                    └───────────────────────────┘
                                  ▲
                                  │
                    ┌───────────────────────────┐
                    │ Tabular Embedding (16-D)  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │       Metadata MLP        │
                    └─────────────┬─────────────┘
                                  │
                                  ▲
                    ┌─────────────┴─────────────┐
                    │ Clinical Metadata (4-D)   │
                    │   Age, Sex, Height, Weight│
                    └───────────────────────────┘
```

### 1.1 ECG Waveform processing Branch (1D-ResNet)
The ECG waveform branch accepts filtered time-series data of shape `(Batch, 12, 1000)`. It features a 1D Residual Network containing:
- **Prep Convolution Block**: 1D Convolution with a kernel size of 15 and stride of 1, mapping 12 input leads to 32 feature channels. Includes Batch Normalization and ReLU.
- **Residual Blocks**: Three sequential blocks with skip connections:
  - **Block 1**: In: 32 channels, Out: 32 channels. Shortcut downsamples time dimension by MaxPool1D (factor of 2).
  - **Block 2**: In: 32 channels, Out: 64 channels. Conv1D matches shortcut dimension changes.
  - **Block 3**: In: 64 channels, Out: 64 channels. 
- **Global Average Pooling (GAP)**: Collapses the temporal sequence dimension to output a flat 64-dimensional feature representation $h_{sig} \in \mathbb{R}^{64}$.

### 1.2 Clinical Demographics Branch (Metadata MLP)
Demographics (Age, Sex, Height, Weight) are preprocessed to shape `(Batch, 4)` and passed to a Multi-Layer Perceptron (MLP):
- **Layer 1**: Linear projection mapping 4 inputs to 32 dimensions, followed by 1D Batch Normalization and ReLU activation.
- **Layer 2**: Linear layer mapping 32 dimensions to a final 16-dimensional tabular demographic representation $h_{meta} \in \mathbb{R}^{16}$.

---

## 2. Adaptive Multi-Modal Fusion Modules

To integrate time-series and demographic characteristics, the architecture supports five config-driven fusion approaches:

### 2.1 Late Fusion Baseline (Concatenation)
Both embeddings are concatenated along the feature dimension:
$$h_{fused} = [h_{sig} \mathbin{\Vert} h_{meta}] \in \mathbb{R}^{80}$$
This representation is mapped directly to the classifier head.

### 2.2 Gated Fusion
Gated fusion learns a dynamic coefficient vector to regulate feature weights:
$$g = \sigma(W_g [h_{sig} \mathbin{\Vert} h_{meta}] + b_g) \in \mathbb{R}^{64}$$
$$h_{fused} = g \odot W_s h_{sig} + (1 - g) \odot W_m h_{meta}$$
where $W_s, W_m, W_g$ are linear transformations and $\sigma$ is the Sigmoid activation.

### 2.3 Cross-Attention Fusion
Utilizes demographic vectors to selectively query and attend signal waveforms.
- **Queries (Q)**: Linear projection of the signal embedding $h_{sig}$.
- **Keys (K) & Values (V)**: Linear projections of the clinical metadata embedding $h_{meta}$.
- Attention scores are calculated via scaled dot-product and mapped back using a skip connection:
$$\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_{model}}}\right) V$$
$$h_{fused} = W_{out} \text{Attention}(Q, K, V) + Q$$

### 2.4 Feature Attention Fusion
Concatenates signal and metadata vectors, then applies self-attention across the combined 80-dimensional space:
$$h_{concat} = [h_{sig} \mathbin{\Vert} h_{meta}] \in \mathbb{R}^{80}$$
$$h_{fused} = \text{Sigmoid}\left(\frac{W_q h_{concat} \cdot W_k h_{concat}}{\sqrt{80}}\right) \odot W_v h_{concat} + h_{concat}$$

### 2.5 Dynamic Weighted Fusion
Learns scalar coefficients to weight the overall contribution of each modality:
$$\alpha = \text{Softmax}(W_w [h_{sig} \mathbin{\Vert} h_{meta}]) \in \mathbb{R}^2$$
$$h_{fused} = \alpha_1 W_s h_{sig} + \alpha_2 W_m h_{meta}$$

---

## 3. Decision Classifier Head
The final representation $h_{fused}$ is projected via a linear layer to a 5-dimensional vector containing predicted diagnostic probabilities for the PTB-XL superclasses:
- Normal ECG (`NORM`)
- Myocardial Infarction (`MI`)
- ST/T Changes (`STTC`)
- Conduction Disturbance (`CD`)
- Hypertrophy (`HYP`)

Multi-label classification utilizes independent `Sigmoid` activation functions for each output node.
