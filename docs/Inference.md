# Model Inference, Uncertainty, & Explainability

This document details the inference pipeline, local explainability generation, prediction uncertainty flagging, and ONNX model deployment.

---

## 1. Probabilistic Inference & Uncertainty Estimation

To provide confidence scores in clinical scenarios, inference integrates **Monte Carlo (MC) Dropout**:
- During forward passes, PyTorch `Dropout` layers are forced to remain active.
- For each input sample, the network runs $N = 30$ times to generate a distribution of predictions.
- **Mean Probabilities**: The final prediction represents the average probability across all MC iterations.
- **Uncertainty Score (Entropy)**: Shannon entropy is calculated across classes:
$$H = - \frac{1}{C} \sum_{c=1}^C [p_c \log_2(p_c) + (1-p_c) \log_2(1-p_c)]$$
- **Uncertainty Flagging**: If the entropy $H$ exceeds $0.75$, the model flags the prediction as "Low Confidence Prediction (Uncertain)", notifying clinical staff that manual review is required.

---

## 2. Explainable AI (XAI) Generation

Explainability maps are generated automatically for diagnostic outputs:
1. **ECG Signal Attention (Grad-CAM)**: Captures features from the 1D prep convolution layer. Weights are averaged across time, highlighting temporal regions on the ECG.
2. **ECG Lead Attribution (Integrated Gradients)**: Evaluates input attribution relative to a flat baseline (zero signal). It returns the contribution of each of the 12 leads over time.
3. **Metadata Attribution (Exact Shapley Values)**: Computes the exact Shapley values across the demographic coalition space ($2^4 = 16$ evaluations) using the training distribution as background. This isolates the contribution of age, sex, weight, and height to the prediction.

---

## 3. ONNX Deployment

The model is exported to ONNX format opset 11. It supports dynamic batch sizing:
- **Inputs**:
  - `ecg_signal`: Shape `(batch_size, 12, 1000)`
  - `clinical_metadata`: Shape `(batch_size, 4)`
- **Outputs**:
  - `diagnostic_probabilities`: Shape `(batch_size, 5)`

To load and run the exported model in python using ONNX Runtime:
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/late_fusion_network.onnx")
dummy_signal = np.zeros((1, 12, 1000), dtype=np.float32)
dummy_metadata = np.zeros((1, 4), dtype=np.float32)

outputs = session.run(None, {
    'ecg_signal': dummy_signal,
    'clinical_metadata': dummy_metadata
})
print("Predicted probabilities:", outputs[0])
```
