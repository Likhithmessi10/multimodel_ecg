import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from scipy import stats
import joblib

from zero_leakage_loader import PTBXLZeroLeakageLoader
from multimodal_fusion_net import SignalOnlyNet, LateFusionNet

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class BCEWithLogitsLossForSigmoidOutput(nn.BCEWithLogitsLoss):
    """
    Subclass of BCEWithLogitsLoss that accepts sigmoid probability outputs.
    It clamps the probabilities and applies the inverse sigmoid (logit) function
    internally before passing to PyTorch's native BCEWithLogitsLoss, preventing
    double-sigmoid distortions while maintaining identical API and training stability.
    """
    def forward(self, input, target):
        eps = 1e-7
        input_clamped = torch.clamp(input, eps, 1.0 - eps)
        logits = torch.log(input_clamped / (1.0 - input_clamped))
        return super().forward(logits, target)

def get_args():
    parser = argparse.ArgumentParser(description="Multi-Modal Cardiac Diagnostics Training Engine")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--subset_size", type=int, default=None, 
                        help="Train on a smaller subset of the dataset (e.g. 2000) for fast iteration")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where data is located")
    parser.add_argument("--models_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--artifacts_dir", type=str, default="./paper_artifacts", help="Directory to save paper artifacts")
    parser.add_argument("--model_type", type=str, default="all", choices=["all", "tabular", "signal", "fusion"],
                        help="Which data branch/model to train: 'tabular' (XGBoost), 'signal' (1D-ResNet), 'fusion' (Late-Fusion Net), or 'all'")
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, criterion, device, is_fusion=True):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        if is_fusion:
            x_wave, x_meta, y = batch
            x_wave, x_meta, y = x_wave.to(device), x_meta.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_wave, x_meta)
        else:
            x_wave, y = batch
            x_wave, y = x_wave.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_wave)
            
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_wave.size(0)
        
    return running_loss / len(dataloader.dataset)

def evaluate_epoch(model, dataloader, criterion, device, is_fusion=True):
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            if is_fusion:
                x_wave, x_meta, y = batch
                x_wave, x_meta, y = x_wave.to(device), x_meta.to(device), y.to(device)
                logits = model(x_wave, x_meta)
            else:
                x_wave, y = batch
                x_wave, y = x_wave.to(device), y.to(device)
                logits = model(x_wave)
                
            loss = criterion(logits, y)
            running_loss += loss.item() * x_wave.size(0)
            
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Since model already outputs sigmoid probabilities, we do not apply Sigmoid again.
    all_probs = all_logits
    return val_loss, all_probs, all_targets

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Scikit-learn multi-label metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Robust AUROC & AUPRC Calculation (Skip classes without both positive and negative targets)
    auroc_list = []
    auprc_list = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            auroc_list.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
            auprc_list.append(average_precision_score(y_true[:, i], y_prob[:, i]))
        else:
            # Fallback values if class is completely missing in test subset
            pass
            
    macro_auroc = np.mean(auroc_list) if len(auroc_list) > 0 else 0.5
    macro_auprc = np.mean(auprc_list) if len(auprc_list) > 0 else 0.5
    
    return {
        "Macro_F1": macro_f1,
        "Micro_F1": micro_f1,
        "Macro_AUROC": macro_auroc,
        "Macro_AUPRC": macro_auprc
    }

def compute_sample_wise_bce(y_true, y_prob):
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    bce = - (y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return np.mean(bce, axis=1) # Average loss across the 5 superclasses for each sample

def run_mcnemar_test(y_true, y_pred_sig, y_pred_fus):
    # Flatten across classes to get aggregate binary correctness
    y_true_f = y_true.flatten()
    pred_sig_f = (y_pred_sig >= 0.5).astype(int).flatten()
    pred_fus_f = (y_pred_fus >= 0.5).astype(int).flatten()
    
    sig_correct = (pred_sig_f == y_true_f)
    fus_correct = (pred_fus_f == y_true_f)
    
    # Contingency Table:
    #                 Fused Correct    Fused Incorrect
    # Signal Correct        a                b
    # Signal Incorrect      c                d
    b = np.sum(sig_correct & ~fus_correct)
    c = np.sum(~sig_correct & fus_correct)
    
    if (b + c) == 0:
        stat = 0.0
        p_val = 1.0
    else:
        stat = (abs(b - c) - 1)**2 / (b + c) # Continuity corrected
        p_val = stats.chi2.sf(stat, df=1)
        
    return stat, p_val

def save_loss_curves(train_losses, val_losses, filepath):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", color="#ff7f0e", linewidth=2)
    plt.title("Convergence Curve: Training vs. Validation Loss (Late-Fusion Network)", fontsize=12, fontweight='bold')
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("BCE Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def save_roc_curves(y_true, y_prob, classes, filepath):
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, cls_name in enumerate(classes):
        if len(np.unique(y_true[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            auc_val = roc_auc_score(y_true[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, color=colors[i], label=f"{cls_name} (AUC = {auc_val:.4f})", linewidth=2)
        else:
            plt.plot([0, 1], [0.5, 0.5], color=colors[i], linestyle=":", label=f"{cls_name} (N/A - Constant Label)", linewidth=1.5, alpha=0.6)
        
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.5)
    plt.title("Multi-Class ROC Curves on Independent Test Set (Fold 10)", fontsize=12, fontweight='bold')
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def save_pr_curves(y_true, y_prob, classes, filepath):
    plt.figure(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, cls_name in enumerate(classes):
        if len(np.unique(y_true[:, i])) > 1:
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
            ap_val = average_precision_score(y_true[:, i], y_prob[:, i])
            plt.plot(recall, precision, color=colors[i], label=f"{cls_name} (AUPRC = {ap_val:.4f})", linewidth=2)
        else:
            plt.plot([0, 1], [0.5, 0.5], color=colors[i], linestyle=":", label=f"{cls_name} (N/A - Constant Label)", linewidth=1.5, alpha=0.6)
        
    plt.title("Multi-Class Precision-Recall Curves on Test Set (Fold 10)", fontsize=12, fontweight='bold')
    plt.xlabel("Recall", fontsize=10)
    plt.ylabel("Precision", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def main():
    args = get_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using device: {device}")
    
    # If on CPU and subset_size is not specified, run on a fast subset to avoid timing out.
    subset_size = args.subset_size
    if device.type == 'cpu' and subset_size is None:
        subset_size = 5000
        print(f"CUDA not available. Subsampling PTB-XL to {subset_size} records to ensure fast execution.")
        
    # Initialize Loader
    loader = PTBXLZeroLeakageLoader(data_dir=args.data_dir)
    
    # Get Splits
    print("Loading data splits...")
    (
        (X_train_wave, X_train_meta, y_train),
        (X_val_wave, X_val_meta, y_val),
        (X_test_wave, X_test_meta, y_test)
    ) = loader.get_data_splits(subset_size=subset_size)
    
    print(f"Train size: {X_train_wave.shape[0]}, Val size: {X_val_wave.shape[0]}, Test size: {X_test_wave.shape[0]}")
    
    # ----------------------------------------------------
    # BASELINE 1: Tabular-Only Classifier (XGBoost)
    # ----------------------------------------------------
    xgb_model = None
    y_prob_xgb = None
    xgb_metrics = None
    xgb_path = os.path.join(args.models_dir, "tabular_xgb_model.joblib")
    
    if args.model_type in ["all", "tabular"]:
        print("\n--- Training Tabular-Only Baseline (XGBoost) ---")
        xgb_estimator = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model = MultiOutputClassifier(xgb_estimator)
        xgb_model.fit(X_train_meta, y_train)
        
        # Predict on test
        xgb_prob_list = xgb_model.predict_proba(X_test_meta)
        y_prob_xgb = np.column_stack([p[:, 1] for p in xgb_prob_list])
        xgb_metrics = compute_metrics(y_test, y_prob_xgb)
        print("XGBoost Baseline Test Metrics:", xgb_metrics)
        
        # Save Tabular Model
        joblib.dump(xgb_model, xgb_path)
    elif os.path.exists(xgb_path):
        print("\n--- Loading Tabular-Only Baseline (XGBoost) from disk ---")
        xgb_model = joblib.load(xgb_path)
        xgb_prob_list = xgb_model.predict_proba(X_test_meta)
        y_prob_xgb = np.column_stack([p[:, 1] for p in xgb_prob_list])
        xgb_metrics = compute_metrics(y_test, y_prob_xgb)
        print("Loaded XGBoost Baseline Test Metrics:", xgb_metrics)
    else:
        print("\n--- Tabular-Only Baseline (XGBoost) not trained or saved ---")
        xgb_metrics = {"Macro_F1": 0.0, "Micro_F1": 0.0, "Macro_AUROC": 0.5, "Macro_AUPRC": 0.5}
        y_prob_xgb = np.zeros_like(y_test)
        
    # ----------------------------------------------------
    # PyTorch Data Loaders
    # ----------------------------------------------------
    # Conver to tensors
    t_train_wave = torch.tensor(X_train_wave, dtype=torch.float32)
    t_train_meta = torch.tensor(X_train_meta, dtype=torch.float32)
    t_train_y = torch.tensor(y_train, dtype=torch.float32)
    
    t_val_wave = torch.tensor(X_val_wave, dtype=torch.float32)
    t_val_meta = torch.tensor(X_val_meta, dtype=torch.float32)
    t_val_y = torch.tensor(y_val, dtype=torch.float32)
    
    t_test_wave = torch.tensor(X_test_wave, dtype=torch.float32)
    t_test_meta = torch.tensor(X_test_meta, dtype=torch.float32)
    t_test_y = torch.tensor(y_test, dtype=torch.float32)
    
    # Create DataLoaders
    # Fusion Loaders
    train_fusion_ds = TensorDataset(t_train_wave, t_train_meta, t_train_y)
    val_fusion_ds = TensorDataset(t_val_wave, t_val_meta, t_val_y)
    test_fusion_ds = TensorDataset(t_test_wave, t_test_meta, t_test_y)
    
    train_fusion_loader = DataLoader(train_fusion_ds, batch_size=args.batch_size, shuffle=True)
    val_fusion_loader = DataLoader(val_fusion_ds, batch_size=args.batch_size, shuffle=False)
    test_fusion_loader = DataLoader(test_fusion_ds, batch_size=args.batch_size, shuffle=False)
    
    # Signal-Only Loaders
    train_sig_ds = TensorDataset(t_train_wave, t_train_y)
    val_sig_ds = TensorDataset(t_val_wave, t_val_y)
    test_sig_ds = TensorDataset(t_test_wave, t_test_y)
    
    train_sig_loader = DataLoader(train_sig_ds, batch_size=args.batch_size, shuffle=True)
    val_sig_loader = DataLoader(val_sig_ds, batch_size=args.batch_size, shuffle=False)
    test_sig_loader = DataLoader(test_sig_ds, batch_size=args.batch_size, shuffle=False)
    
    criterion = BCEWithLogitsLossForSigmoidOutput()
    
    # ----------------------------------------------------
    # BASELINE 2: Signal-Only Classifier (1D-ResNet)
    # ----------------------------------------------------
    sig_model = None
    y_prob_sig_test = None
    sig_metrics = None
    sig_path = os.path.join(args.models_dir, "signal_only_model.pth")
    
    if args.model_type in ["all", "signal"]:
        print("\n--- Training Signal-Only Baseline (1D-ResNet) ---")
        sig_model = SignalOnlyNet(in_channels=12, num_classes=5).to(device)
        optimizer_sig = optim.Adam(sig_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_sig_state = None
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(sig_model, train_sig_loader, optimizer_sig, criterion, device, is_fusion=False)
            val_loss, _, _ = evaluate_epoch(sig_model, val_sig_loader, criterion, device, is_fusion=False)
            
            print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_sig_state = {k: v.cpu() for k, v in sig_model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        # Load best weights and evaluate on test
        sig_model.load_state_dict({k: v.to(device) for k, v in best_sig_state.items()})
        torch.save(sig_model.state_dict(), sig_path)
        
        _, y_prob_sig_test, _ = evaluate_epoch(sig_model, test_sig_loader, criterion, device, is_fusion=False)
        sig_metrics = compute_metrics(y_test, y_prob_sig_test)
        print("Signal-Only Baseline Test Metrics:", sig_metrics)
    elif os.path.exists(sig_path):
        print("\n--- Loading Signal-Only Baseline (1D-ResNet) from disk ---")
        sig_model = SignalOnlyNet(in_channels=12, num_classes=5).to(device)
        sig_model.load_state_dict(torch.load(sig_path, map_location=device))
        _, y_prob_sig_test, _ = evaluate_epoch(sig_model, test_sig_loader, criterion, device, is_fusion=False)
        sig_metrics = compute_metrics(y_test, y_prob_sig_test)
        print("Loaded Signal-Only Baseline Test Metrics:", sig_metrics)
    else:
        print("\n--- Signal-Only Baseline (1D-ResNet) not trained or saved ---")
        sig_metrics = {"Macro_F1": 0.0, "Micro_F1": 0.0, "Macro_AUROC": 0.5, "Macro_AUPRC": 0.5}
        y_prob_sig_test = np.zeros_like(y_test)
        
    # ----------------------------------------------------
    # SYSTEM 3: Proposed Multi-Modal Late-Fusion Network
    # ----------------------------------------------------
    fusion_model = None
    y_prob_fus_test = None
    fusion_metrics = None
    fusion_train_losses = []
    fusion_val_losses = []
    fusion_path = os.path.join(args.models_dir, "late_fusion_model.pth")
    
    if args.model_type in ["all", "fusion"]:
        print("\n--- Training Proposed Multi-Modal Late-Fusion Network ---")
        fusion_model = LateFusionNet(in_channels=12, meta_features=4, num_classes=5).to(device)
        optimizer_fus = optim.Adam(fusion_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_fus_state = None
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(fusion_model, train_fusion_loader, optimizer_fus, criterion, device, is_fusion=True)
            val_loss, _, _ = evaluate_epoch(fusion_model, val_fusion_loader, criterion, device, is_fusion=True)
            
            fusion_train_losses.append(train_loss)
            fusion_val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_fus_state = {k: v.cpu() for k, v in fusion_model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        # Load best weights and evaluate on test
        fusion_model.load_state_dict({k: v.to(device) for k, v in best_fus_state.items()})
        torch.save(fusion_model.state_dict(), fusion_path)
        
        _, y_prob_fus_test, _ = evaluate_epoch(fusion_model, test_fusion_loader, criterion, device, is_fusion=True)
        fusion_metrics = compute_metrics(y_test, y_prob_fus_test)
        print("Multi-Modal Late-Fusion Test Metrics:", fusion_metrics)
    elif os.path.exists(fusion_path):
        print("\n--- Loading Proposed Multi-Modal Late-Fusion Network from disk ---")
        fusion_model = LateFusionNet(in_channels=12, meta_features=4, num_classes=5).to(device)
        fusion_model.load_state_dict(torch.load(fusion_path, map_location=device))
        _, y_prob_fus_test, _ = evaluate_epoch(fusion_model, test_fusion_loader, criterion, device, is_fusion=True)
        fusion_metrics = compute_metrics(y_test, y_prob_fus_test)
        print("Loaded Proposed Multi-Modal Late-Fusion Network Test Metrics:", fusion_metrics)
    else:
        print("\n--- Proposed Multi-Modal Late-Fusion Network not trained or saved ---")
        fusion_metrics = {"Macro_F1": 0.0, "Micro_F1": 0.0, "Macro_AUROC": 0.5, "Macro_AUPRC": 0.5}
        y_prob_fus_test = np.zeros_like(y_test)
        
    # ----------------------------------------------------
    # STAGE 2: MANDATORY PAPER ARTIFACT EXPORT
    # ----------------------------------------------------
    print("\n--- Exporting Paper Artifacts to ./paper_artifacts/ ---")
    
    # TABLE 1: Hyperparameters
    hyperparams = {
        "Hyperparameter": [
            "Learning Rate (LR)",
            "Weight Decay",
            "Batch Size",
            "Optimizer Selection",
            "Early Stopping Patience (Val Loss)",
            "Deep Learning Epoch Length (Max)",
            "Deep Learning Epochs Run",
            "Signal 1D CNN Residual Blocks",
            "Signal Embedding Size (GAP)",
            "Metadata Input Features",
            "Metadata MLP Layer Sizes",
            "Metadata Embedding Size",
            "Late Fusion Embedding Junction",
            "Target Superclass Nodes",
            "Loss Function Objective",
            "Tabular Classifier",
            "Tabular Estimators"
        ],
        "Value": [
            str(args.lr),
            str(args.weight_decay),
            str(args.batch_size),
            "Adam",
            str(args.patience),
            str(args.epochs),
            str(len(fusion_train_losses)) if len(fusion_train_losses) > 0 else "N/A (Loaded from checkpoint)",
            "3 Blocks (1D Conv + BN + ReLU + MaxPool1D)",
            "64",
            "4 (age, sex, height, weight)",
            "4 -> 32 -> 16",
            "16",
            "80 (64 signal + 16 metadata)",
            "5 (NORM, MI, STTC, CD, HYP)",
            "BCEWithLogitsLoss (Multi-label)",
            "XGBoost (MultiOutputClassifier)",
            "100 estimators, max_depth=5, lr=0.05"
        ]
    }
    df_table1 = pd.DataFrame(hyperparams)
    df_table1.to_csv(os.path.join(args.artifacts_dir, "table1_hyperparameters.csv"), index=False)
    
    # TABLE 2: Ablation Results
    ablation_data = {
        "Configuration": ["Tabular-Only Baseline (XGBoost)", "Signal-Only Baseline (1D-ResNet)", "Proposed Late-Fusion Network"],
        "Macro F1": [xgb_metrics["Macro_F1"], sig_metrics["Macro_F1"], fusion_metrics["Macro_F1"]],
        "Micro F1": [xgb_metrics["Micro_F1"], sig_metrics["Micro_F1"], fusion_metrics["Micro_F1"]],
        "Macro AUROC": [xgb_metrics["Macro_AUROC"], sig_metrics["Macro_AUROC"], fusion_metrics["Macro_AUROC"]],
        "Macro AUPRC": [xgb_metrics["Macro_AUPRC"], sig_metrics["Macro_AUPRC"], fusion_metrics["Macro_AUPRC"]]
    }
    df_table2 = pd.DataFrame(ablation_data)
    df_table2.to_csv(os.path.join(args.artifacts_dir, "table2_ablation_results.csv"), index=False)
    
    # TABLE 3: Statistical Significance
    # Compute paired test-statistic comparing Fused Predictions directly against Signal-Only predictions
    loss_sig_sample = compute_sample_wise_bce(y_test, y_prob_sig_test)
    loss_fus_sample = compute_sample_wise_bce(y_test, y_prob_fus_test)
    
    # Paired t-test on BCE loss per sample
    t_stat_loss, p_val_loss = stats.ttest_rel(loss_sig_sample, loss_fus_sample)
    
    # McNemar's test on prediction correctness
    mcnemar_stat, p_val_mcnemar = run_mcnemar_test(y_test, y_prob_sig_test, y_prob_fus_test)
    
    stat_data = {
        "Statistical Metric": [
            "Paired t-test statistic (BCE Loss)", 
            "Paired t-test p-value (BCE Loss)",
            "McNemar's test statistic (Accuracy)",
            "McNemar's test p-value (Accuracy)"
        ],
        "Value": [t_stat_loss, p_val_loss, mcnemar_stat, p_val_mcnemar],
        "Interpretation": [
            "Difference in BCE loss distribution",
            "Stat significance of loss improvement (p < 0.05 is significant)",
            "Correctness contingency difference",
            "Stat significance of prediction accuracy changes"
        ]
    }
    df_table3 = pd.DataFrame(stat_data)
    df_table3.to_csv(os.path.join(args.artifacts_dir, "table3_statistical_significance.csv"), index=False)
    
    # Save Tables in Markdown format for the human researcher
    with open(os.path.join(args.artifacts_dir, "README.md"), "w") as f:
        f.write("# PTB-XL Late-Fusion Project: Academic Paper Artifacts\n\n")
        f.write("This directory contains exact empirical results, statistical validation, and figures generated from training and testing configurations on PTB-XL (Fold 10 Test Set).\n\n")
        
        f.write("## Table 1: Mathematical Hyperparameters\n")
        f.write(df_table1.to_markdown(index=False) + "\n\n")
        
        f.write("## Table 2: Ablation Results\n")
        f.write(df_table2.to_markdown(index=False) + "\n\n")
        
        f.write("## Table 3: Statistical Significance Validation\n")
        f.write(df_table3.to_markdown(index=False) + "\n\n")
        
        f.write("## Generated High-Resolution Figures\n")
        f.write("- **Loss Curves**: `figure1_loss_curves.png` (Training convergence)\n")
        f.write("- **ROC Curves**: `figure2_roc_curves.png` (Authoritative class sensitivities)\n")
        f.write("- **PR Curves**: `figure3_pr_curves.png` (Imbalanced clinical evaluation)\n")
        
    # Generate High-Resolution Visualizations
    if len(fusion_train_losses) > 0:
        save_loss_curves(
            fusion_train_losses, 
            fusion_val_losses, 
            os.path.join(args.artifacts_dir, "figure1_loss_curves.png")
        )
    
    if y_prob_fus_test is not None and not np.all(y_prob_fus_test == 0.0):
        save_roc_curves(
            y_test, 
            y_prob_fus_test, 
            loader.superclasses, 
            os.path.join(args.artifacts_dir, "figure2_roc_curves.png")
        )
        
        save_pr_curves(
            y_test, 
            y_prob_fus_test, 
            loader.superclasses, 
            os.path.join(args.artifacts_dir, "figure3_pr_curves.png")
        )
    
    print("Paper artifacts generated successfully!")

if __name__ == "__main__":
    main()
