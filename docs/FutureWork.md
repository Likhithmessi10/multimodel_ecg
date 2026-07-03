# Future Research Directions

This document details future directions and extensions.

---

## 1. Architectural and Fusion extensions
1. **Multimodal Transformers**: Implementing unified multi-modal Transformers (e.g., Perceiver or cross-attention encoders) to model interactions between raw signals and demographics.
2. **Dynamic Fusion Modules**: Exploring advanced fusion modules such as Bilinear Pooling or Tensor Fusion Networks to capture high-order modal relationships.

---

## 2. Dataset and Modality expansion
1. **Clinical Narrative Integration**: Integrating raw text from ECG clinical notes and reports using pre-trained medical language models (e.g., ClinicalBERT).
2. **External Validation Folds**: Evaluating generalizability by testing on other public repositories, such as the PhysioNet/CinC Challenge datasets or private clinical registries.

---

## 3. Deployment and Optimization
1. **Explainable AI Integration**: Integrating Grad-CAM and Integrated Gradients overlays directly into Streamlit-based dashboards for clinicians.
2. **Hardware Acceleration**: Quantizing the exported ONNX model (FP16 or INT8 precision) to evaluate latency reductions on edge micro-controllers and mobile health monitors.
