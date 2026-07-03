# PTB-XL Late-Fusion Project: Academic Paper Artifacts

This directory contains exact empirical results, statistical validation, and figures generated from training and testing configurations on PTB-XL (Fold 10 Test Set).

## Table 1: Mathematical Hyperparameters
| Hyperparameter                     | Value                                      |
|:-----------------------------------|:-------------------------------------------|
| Learning Rate (LR)                 | 0.001                                      |
| Weight Decay                       | 0.0001                                     |
| Batch Size                         | 64                                         |
| Optimizer Selection                | Adam                                       |
| Early Stopping Patience (Val Loss) | 5                                          |
| Deep Learning Epoch Length (Max)   | 15                                         |
| Deep Learning Epochs Run           | N/A (Loaded from checkpoint)               |
| Signal 1D CNN Residual Blocks      | 3 Blocks (1D Conv + BN + ReLU + MaxPool1D) |
| Signal Embedding Size (GAP)        | 64                                         |
| Metadata Input Features            | 4 (age, sex, height, weight)               |
| Metadata MLP Layer Sizes           | 4 -> 32 -> 16                              |
| Metadata Embedding Size            | 16                                         |
| Late Fusion Embedding Junction     | 80 (64 signal + 16 metadata)               |
| Target Superclass Nodes            | 5 (NORM, MI, STTC, CD, HYP)                |
| Loss Function Objective            | BCEWithLogitsLoss (Multi-label)            |
| Tabular Classifier                 | XGBoost (MultiOutputClassifier)            |
| Tabular Estimators                 | 100 estimators, max_depth=5, lr=0.05       |

## Table 2: Ablation Results
| Configuration                    |   Macro F1 |   Micro F1 |   Macro AUROC |   Macro AUPRC |
|:---------------------------------|-----------:|-----------:|--------------:|--------------:|
| Tabular-Only Baseline (XGBoost)  |   0.341935 |   0.666667 |      0.55326  |      0.497704 |
| Signal-Only Baseline (1D-ResNet) |   0.196825 |   0.566667 |      0.572368 |      0.492962 |
| Proposed Late-Fusion Network     |   0.334641 |   0.6      |      0.790136 |      0.527844 |

## Table 3: Statistical Significance Validation
| Statistical Metric                  |    Value | Interpretation                                                  |
|:------------------------------------|---------:|:----------------------------------------------------------------|
| Paired t-test statistic (BCE Loss)  | 0.484307 | Difference in BCE loss distribution                             |
| Paired t-test p-value (BCE Loss)    | 0.633704 | Stat significance of loss improvement (p < 0.05 is significant) |
| McNemar's test statistic (Accuracy) | 0.125    | Correctness contingency difference                              |
| McNemar's test p-value (Accuracy)   | 0.723674 | Stat significance of prediction accuracy changes                |

## Generated High-Resolution Figures
- **Loss Curves**: `figure1_loss_curves.png` (Training convergence)
- **ROC Curves**: `figure2_roc_curves.png` (Authoritative class sensitivities)
- **PR Curves**: `figure3_pr_curves.png` (Imbalanced clinical evaluation)
