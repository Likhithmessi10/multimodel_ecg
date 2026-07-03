# Research Limitations

This document discusses the limitations of the current framework.

---

## 1. Modality and demographic constraints
1. **Limited Demographic Attributes**: The patient metadata consists of only four static variables: age, sex, weight, and height. It lacks clinical context such as patient history, comorbidities, lab values, or current medications.
2. **Binary Sex Representation**: Sex is modeled as a binary variable (Male/Female), which does not represent non-binary cohorts or detailed clinical gender profiles.

---

## 2. Signal and algorithmic limits
1. **Fixed Sampling Rate**: Waveforms are processed at a fixed sampling rate of 100 Hz. While this reduces computational overhead, it may discard high-frequency characteristics detectable at 500 Hz.
2. **Fixed Time-Series Length**: The model requires input waveforms to be exactly 10 seconds long (1000 samples). It does not dynamically process variable-length ECG recordings without truncation or zero-padding.
3. **Exact Shapley Computation**: Exact Shapley values are calculated across demographic features. While efficient for 4 features ($2^4 = 16$ evaluations), this approach does not scale to datasets with many clinical variables.

---

## 3. Validation limits
1. **Single-Center Focus**: The model is trained and evaluated using the PTB-XL dataset. External validation on separate datasets from different hospital systems is required to evaluate generalizability across different recording devices and demographics.
