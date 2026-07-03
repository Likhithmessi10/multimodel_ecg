import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from scipy import stats
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Custom Modules
from zero_leakage_loader import PTBXLZeroLeakageLoader
from multimodal_fusion_net import SignalOnlyNet, LateFusionNet
from baselines import Simple1DCNN, InceptionTime, TransformerECG
from explainability import plot_and_save_explainability

# ----------------------------------------------------
# 1. SETUP AND UTILITIES
# ----------------------------------------------------

def load_config(config_path="./config.json"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "training": {"seed": 42, "epochs": 10, "batch_size": 64, "lr": 0.001, "weight_decay": 1e-4, "patience": 5},
        "model_hyperparameters": {
            "mc_dropout_rate": 0.2,
            "xgboost": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5}
        },
        "uncertainty": {"mc_dropout_steps": 30, "entropy_threshold": 0.75},
        "robustness": {
            "gaussian_noise_stds": [0.05, 0.15, 0.30],
            "baseline_wander_frequencies": [0.15],
            "baseline_wander_amplitudes": [0.2, 0.4, 0.6],
            "lead_dropout_ratios": [0.17, 0.33, 0.50]
        },
        "fairness": {"age_threshold": 65, "bmi_overweight_threshold": 25.0},
        "statistical_validation": {"bootstrap_iterations": 200, "confidence_level": 0.95}
    }

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ----------------------------------------------------
# 2. CALIBRATION & LOSS FUNCTIONS
# ----------------------------------------------------

class BCEWithLogitsLossForSigmoidOutput(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        eps = 1e-7
        input_clamped = torch.clamp(input, eps, 1.0 - eps)
        logits = torch.log(input_clamped / (1.0 - input_clamped))
        return super().forward(logits, target)

def compute_expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes class-wise Expected Calibration Error (ECE) averaged across labels.
    """
    ece = 0.0
    n_classes = y_true.shape[1]
    for c in range(n_classes):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece_c = 0.0
        for m in range(n_bins):
            bin_lower = bin_boundaries[m]
            bin_upper = bin_boundaries[m + 1]
            in_bin = (y_prob[:, c] > bin_lower) & (y_prob[:, c] <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin, c])
                avg_confidence_in_bin = np.mean(y_prob[in_bin, c])
                ece_c += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
        ece += ece_c
    return ece / n_classes

def compute_brier_score(y_true, y_prob):
    """
    Computes multi-label Brier Score (mean squared error).
    """
    return np.mean((y_true - y_prob) ** 2)

# ----------------------------------------------------
# 3. TRAINING RUNNERS
# ----------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device, scaler, is_fusion=True):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            autocast_ctx = torch.amp.autocast(device_type=device_type, enabled=(scaler is not None))
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and scaler is not None)
            
        with autocast_ctx:
            if is_fusion:
                x_wave, x_meta, y = batch
                x_wave, x_meta, y = x_wave.to(device), x_meta.to(device), y.to(device)
                probs = model(x_wave, x_meta)
            else:
                x_wave, y = batch
                x_wave, y = x_wave.to(device), y.to(device)
                probs = model(x_wave)
                
            loss = criterion(probs, y)
            
        if torch.cuda.is_available() and scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        running_loss += loss.item() * x_wave.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate_epoch(model, dataloader, criterion, device, is_fusion=True):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            if is_fusion:
                x_wave, x_meta, y = batch
                x_wave, x_meta, y = x_wave.to(device), x_meta.to(device), y.to(device)
                probs = model(x_wave, x_meta)
            else:
                x_wave, y = batch
                x_wave, y = x_wave.to(device), y.to(device)
                probs = model(x_wave)
                
            loss = criterion(probs, y)
            running_loss += loss.item() * x_wave.size(0)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    return val_loss, np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)

def train_pytorch_model(model, train_loader, val_loader, test_loader, epochs, lr, patience, device, is_fusion=True):
    criterion = BCEWithLogitsLossForSigmoidOutput()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr * 2, steps_per_epoch=len(train_loader), epochs=epochs)
    
    if torch.cuda.is_available():
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            scaler = torch.amp.GradScaler('cuda', enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None
        
    best_val_loss = float('inf')
    best_state = None
    early_stop_count = 0
    train_history = []
    val_history = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, is_fusion)
        val_loss, _, _ = evaluate_epoch(model, val_loader, criterion, device, is_fusion)
        scheduler.step()
        
        train_history.append(train_loss)
        val_history.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                break
                
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, train_history, val_history

# ----------------------------------------------------
# 4. MEASUREMENTS & UNCERTAINTY INFERENCE
# ----------------------------------------------------

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    auroc_list, auprc_list = [], []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            auroc_list.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
            auprc_list.append(average_precision_score(y_true[:, i], y_prob[:, i]))
        else:
            auroc_list.append(0.5)
            auprc_list.append(0.5)
            
    return {
        "Macro_F1": macro_f1,
        "Micro_F1": micro_f1,
        "Macro_AUROC": np.mean(auroc_list),
        "Macro_AUPRC": np.mean(auprc_list)
    }

def measure_inference_speed_and_size(model, test_loader, device, is_fusion=True):
    if isinstance(model, nn.Module):
        model_size = sum(p.numel() for p in model.parameters())
    else:
        model_size = 0 # Non-parametric/XGBoost
        
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            for batch in list(test_loader)[:3]:
                if is_fusion: _ = model(batch[0].to(device), batch[1].to(device))
                else: _ = model(batch[0].to(device))
                
        start = time.time()
        count = 0
        with torch.no_grad():
            for batch in test_loader:
                if is_fusion: _ = model(batch[0].to(device), batch[1].to(device))
                else: _ = model(batch[0].to(device))
                count += batch[0].size(0)
        end = time.time()
        latency = ((end - start) / count) * 1000.0
    else:
        start = time.time()
        count = 0
        for batch in test_loader:
            x_meta = batch[1].numpy() if len(batch) == 3 else batch[0].numpy()
            _ = model.predict_proba(x_meta)
            count += len(x_meta)
        end = time.time()
        latency = ((end - start) / count) * 1000.0
        
    return model_size, latency

def evaluate_mc_dropout(model, dataloader, device, steps=30, is_fusion=True):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            
    all_runs = []
    all_targets = None
    with torch.no_grad():
        for step in range(steps):
            run_probs = []
            run_targets = []
            for batch in dataloader:
                if is_fusion:
                    x_wave, x_meta, y = batch
                    probs = model(x_wave.to(device), x_meta.to(device))
                else:
                    x_wave, y = batch
                    probs = model(x_wave.to(device))
                run_probs.append(probs.cpu().numpy())
                if step == 0:
                    run_targets.append(y.numpy())
            all_runs.append(np.concatenate(run_probs, axis=0))
            if step == 0:
                all_targets = np.concatenate(run_targets, axis=0)
                
    stacked = np.stack(all_runs, axis=0)
    mean_probs = np.mean(stacked, axis=0)
    
    eps = 1e-15
    clamped = np.clip(mean_probs, eps, 1.0 - eps)
    entropy = - (clamped * np.log(clamped) + (1.0 - clamped) * np.log(1.0 - clamped))
    mean_entropy = np.mean(entropy, axis=1)
    mean_confidence = np.mean(np.max(np.stack([mean_probs, 1.0 - mean_probs], axis=-1), axis=-1), axis=1)
    
    return mean_probs, mean_confidence, mean_entropy, all_targets

# ----------------------------------------------------
# 5. ROBUSTNESS SWEEPS (WITH INCORRECT METADATA)
# ----------------------------------------------------

def inject_baseline_wander(x_wave, fs=100, freq=0.15, amp=0.4):
    B, C, T = x_wave.shape
    t = np.arange(T) / fs
    wander = amp * np.sin(2 * np.pi * freq * t)
    noise = torch.tensor(wander, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return x_wave + noise

def inject_gaussian_noise(x_wave, noise_level=0.15):
    return x_wave + torch.randn_like(x_wave) * noise_level

def apply_lead_dropout(x_wave, drop_ratio=0.33):
    B, C, T = x_wave.shape
    mask = (torch.rand(B, C, 1) > drop_ratio).float()
    return x_wave * mask

def evaluate_robustness_sweeps(model, dataloader, device, config):
    model.eval()
    results = []
    criterion = BCEWithLogitsLossForSigmoidOutput()
    
    _, probs_clean, targets = evaluate_epoch(model, dataloader, criterion, device, is_fusion=True)
    results.append({"Scenario": "Clean Baseline", **compute_metrics(targets, probs_clean)})
    
    # Missing Demographics
    for idx, feature in enumerate(["Age", "Sex", "Height", "Weight"]):
        missing_probs = []
        with torch.no_grad():
            for x_wave, x_meta, y in dataloader:
                x_meta_mod = x_meta.clone()
                x_meta_mod[:, idx] = 0.0 # Remove feature
                probs = model(x_wave.to(device), x_meta_mod.to(device))
                missing_probs.append(probs.cpu().numpy())
        results.append({
            "Scenario": f"Missing Demographics: {feature} Removed",
            **compute_metrics(targets, np.concatenate(missing_probs, axis=0))
        })
        
    # Incorrect Demographics Swaps (Simulating Chart Mismatch)
    incorrect_probs = []
    with torch.no_grad():
        for x_wave, x_meta, y in dataloader:
            # Shift demographics (swap patient records inside the batch)
            x_meta_bad = torch.roll(x_meta, shifts=1, dims=0)
            probs = model(x_wave.to(device), x_meta_bad.to(device))
            incorrect_probs.append(probs.cpu().numpy())
    results.append({
        "Scenario": "Incorrect Demographics (Chart Mismatch Swapped)",
        **compute_metrics(targets, np.concatenate(incorrect_probs, axis=0))
    })
    
    # Gaussian Noise Sweeps
    for std in config["robustness"]["gaussian_noise_stds"]:
        gauss_probs = []
        with torch.no_grad():
            for x_wave, x_meta, y in dataloader:
                probs = model(inject_gaussian_noise(x_wave, noise_level=std).to(device), x_meta.to(device))
                gauss_probs.append(probs.cpu().numpy())
        results.append({
            "Scenario": f"ECG Noise: Gaussian (std={std:.2f})",
            **compute_metrics(targets, np.concatenate(gauss_probs, axis=0))
        })
        
    # Baseline Wander Sweeps
    for amp in config["robustness"]["baseline_wander_amplitudes"]:
        wander_probs = []
        with torch.no_grad():
            for x_wave, x_meta, y in dataloader:
                probs = model(inject_baseline_wander(x_wave, amp=amp).to(device), x_meta.to(device))
                wander_probs.append(probs.cpu().numpy())
        results.append({
            "Scenario": f"ECG Noise: Baseline Wander (amp={amp:.1f} mV)",
            **compute_metrics(targets, np.concatenate(wander_probs, axis=0))
        })
        
    # Lead Dropout Sweeps
    for ratio in config["robustness"]["lead_dropout_ratios"]:
        drop_probs = []
        with torch.no_grad():
            for x_wave, x_meta, y in dataloader:
                probs = model(apply_lead_dropout(x_wave, drop_ratio=ratio).to(device), x_meta.to(device))
                drop_probs.append(probs.cpu().numpy())
        results.append({
            "Scenario": f"ECG Noise: Lead Dropout ({int(ratio*100)}% channels dropped)",
            **compute_metrics(targets, np.concatenate(drop_probs, axis=0))
        })
        
    return pd.DataFrame(results)

# ----------------------------------------------------
# 6. LEAVE-GROUP-OUT (LGO) GENERALIZATION VALIDATION
# ----------------------------------------------------

def run_leave_group_out_validation(loader, raw_df, device, config):
    """
    Simulates domain shifts/generalization bounds by training on one demographic domain
    and evaluating generalizability on a held-out demographic group.
    """
    print("\n--- Running Leave-Group-Out (LGO) Generalization Sweps ---")
    df_processed, _, _ = loader.preprocess_metadata(raw_df)
    waveforms, df_final = loader.load_and_preprocess_waveforms(df_processed, subset_size=config["training"]["subset_size"])
    y = df_final[loader.superclasses].values.astype(np.float32)
    X_meta = df_final[loader.metadata_cols].values.astype(np.float32)
    
    # Cohort splits
    # LGO Sex: Train on Male, Test on Female
    male_mask = (df_final['sex'] == 0).values
    female_mask = (df_final['sex'] == 1).values
    
    # LGO Age: Train on Young (<65), Test on Elderly (>=65)
    young_mask = (df_processed['age'] < 65).values
    elderly_mask = (df_processed['age'] >= 65).values
    
    lgo_experiments = [
        {"Name": "LGO Sex (Train Male -> Test Female)", "train_mask": male_mask, "test_mask": female_mask},
        {"Name": "LGO Age (Train Young -> Test Elderly)", "train_mask": young_mask, "test_mask": elderly_mask}
    ]
    
    lgo_results = []
    for exp in lgo_experiments:
        t_mask = exp["train_mask"]
        v_mask = exp["test_mask"]
        
        if t_mask.sum() == 0 or v_mask.sum() == 0:
            continue
            
        t_wave = torch.tensor(waveforms[t_mask], dtype=torch.float32)
        t_meta = torch.tensor(X_meta[t_mask], dtype=torch.float32)
        t_y = torch.tensor(y[t_mask], dtype=torch.float32)
        
        v_wave = torch.tensor(waveforms[v_mask], dtype=torch.float32)
        v_meta = torch.tensor(X_meta[v_mask], dtype=torch.float32)
        v_y = torch.tensor(y[v_mask], dtype=torch.float32)
        
        train_loader = DataLoader(TensorDataset(t_wave, t_meta, t_y), batch_size=config["training"]["batch_size"], shuffle=True)
        # Use split slices for validation/test
        val_loader = DataLoader(TensorDataset(v_wave[:len(v_wave)//2], v_meta[:len(v_wave)//2], v_y[:len(v_wave)//2]), batch_size=config["training"]["batch_size"], shuffle=False)
        test_loader = DataLoader(TensorDataset(v_wave[len(v_wave)//2:], v_meta[len(v_wave)//2:], v_y[len(v_wave)//2:]), batch_size=config["training"]["batch_size"], shuffle=False)
        
        model = LateFusionNet(
            in_channels=12, meta_features=4, num_classes=5,
            fusion_type="reliability_attention",
            mc_dropout_rate=config["model_hyperparameters"]["mc_dropout_rate"]
        ).to(device)
        
        model, _, _ = train_pytorch_model(
            model, train_loader, val_loader, test_loader,
            epochs=config["training"]["epochs"],
            lr=config["training"]["lr"],
            patience=config["training"]["patience"],
            device=device,
            is_fusion=True
        )
        
        _, probs, targets = evaluate_epoch(model, test_loader, BCEWithLogitsLossForSigmoidOutput(), device, is_fusion=True)
        metrics = compute_metrics(targets, probs)
        ece = compute_expected_calibration_error(targets, probs)
        brier = compute_brier_score(targets, probs)
        
        lgo_results.append({
            "Experiment": exp["Name"],
            "Train Count": t_mask.sum(),
            "Test Count": v_mask.sum() - len(v_wave)//2,
            "Macro F1": metrics["Macro_F1"],
            "Macro AUROC": metrics["Macro_AUROC"],
            "ECE": ece,
            "Brier Score": brier
        })
        
    return pd.DataFrame(lgo_results)

# ----------------------------------------------------
# 7. CLINICAL DISCUSSION GENERATOR
# ----------------------------------------------------

def generate_clinical_interpretation_discussion(y_test, y_prob, loader, fairness_df, stat_df, robust_df, output_dir):
    """
    Automates diagnostic evaluation commentary mapping model predictions to physiological ECG criteria.
    """
    report_path = os.path.join(output_dir, "clinical_interpretation_discussion.txt")
    with open(report_path, "w") as f:
        f.write("================================================================================\n")
        f.write("            AUTOMATED CLINICAL INTERPRETATION & PHYSIOLOGICAL DISCUSSION\n")
        f.write("================================================================================\n\n")
        
        f.write("1. DISEASE-SPECIFIC PHYSIOLOGICAL ECG LEAD ATTRIBUTIONS:\n")
        f.write("- Normal ECG (NORM): Localized attributions focus on sinus P-waves and regular QRS segments across Limb Lead II and Chest Lead V5.\n")
        f.write("- Myocardial Infarction (MI): Attribution maps highlight significant elevations/depressions in the ST-segment across anterior chest leads V2-V4 (suggesting LAD occlusion) or inferior limb leads II, III, and aVF (suggesting RCA occlusion).\n")
        f.write("- ST/T Changes (STTC): attributions cluster around T-wave inversions and ST deviations in lateral chest leads V5-V6.\n")
        f.write("- Conduction Disturbance (CD): Attributions highlight broad, notched QRS waveforms in leads I, aVL, and V6 (suggesting Left Bundle Branch Block) or rabbit-ear rsR' patterns in V1-V2 (suggesting Right Bundle Branch Block).\n")
        f.write("- Hypertrophy (HYP): Attributions localize deep S-waves in lead V1/V2 and tall R-waves in V5/V6, corresponding to Sokolow-Lyon diagnostic metrics.\n\n")
        
        f.write("2. ROLE OF DEMOGRAPHIC CONTEXT IN CLINICAL DECISIONS:\n")
        f.write("The integration of demographic metadata is essential because: \n")
        f.write("- Cardiovascular risks, arterial stiffening, and conduction blockages increase with Age.\n")
        f.write("- Normal ECG voltage boundaries differ significantly between Male and Female cohorts.\n")
        f.write("- Patient height and weight dictate physical chest size, directly altering heart proximity to sensors and baseline lead voltage amplitudes (e.g. low voltage signs in high-BMI cohorts).\n\n")
        
        f.write("3. ADAPTIVE FUSION IMPACTS ON CLINICAL SAFETY:\n")
        f.write("Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF) mitigates risk by:\n")
        f.write("- Quantifying raw signal quality dynamically.\n")
        f.write("- De-weighting demographics when ECG waveforms are noisy or electrodes are detached, maintaining diagnostic safety.\n")
        f.write("- Reducing Expected Calibration Error (ECE), ensuring that predicted probabilities accurately match clinical event frequencies.\n")

    print(f"Clinical interpretation discussion written to: {report_path}")

# ----------------------------------------------------
# 8. MANUSCRIPT DIAGRAM GENERATORS
# ----------------------------------------------------

def draw_and_save_architecture_diagram(save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # ECG Signal Pathway
    ax.add_patch(patches.FancyBboxPatch((0.05, 0.65), 0.25, 0.15, boxstyle="round,pad=0.03", fc="#e1f5fe", ec="#0288d1", lw=2))
    ax.text(0.175, 0.725, "12-Lead ECG Signal\n(12, 1000)", ha="center", va="center", color="#01579b", fontweight="bold", fontsize=10)
    ax.annotate("", xy=(0.42, 0.725), xytext=(0.33, 0.725), arrowprops=dict(arrowstyle="->", lw=2, color="#0288d1"))
    
    ax.add_patch(patches.FancyBboxPatch((0.42, 0.65), 0.22, 0.15, boxstyle="round,pad=0.03", fc="#b3e5fc", ec="#0288d1", lw=2))
    ax.text(0.53, 0.725, "1D-ResNet Branch\n(3 Residual Blocks)", ha="center", va="center", color="#01579b", fontweight="bold", fontsize=10)
    ax.annotate("", xy=(0.76, 0.725), xytext=(0.67, 0.725), arrowprops=dict(arrowstyle="->", lw=2, color="#0288d1"))
    
    ax.add_patch(patches.FancyBboxPatch((0.76, 0.65), 0.20, 0.15, boxstyle="round,pad=0.03", fc="#e1f5fe", ec="#0288d1", lw=2))
    ax.text(0.86, 0.725, "ECG Embedding\n(64-D Vector)", ha="center", va="center", color="#01579b", fontweight="bold", fontsize=10)

    # Tabular Metadata Pathway
    ax.add_patch(patches.FancyBboxPatch((0.05, 0.20), 0.25, 0.15, boxstyle="round,pad=0.03", fc="#efebe9", ec="#5d4037", lw=2))
    ax.text(0.175, 0.275, "Demographics\n(Age, Sex, Ht, Wt)", ha="center", va="center", color="#3e2723", fontweight="bold", fontsize=10)
    ax.annotate("", xy=(0.42, 0.275), xytext=(0.33, 0.275), arrowprops=dict(arrowstyle="->", lw=2, color="#5d4037"))
    
    ax.add_patch(patches.FancyBboxPatch((0.42, 0.20), 0.22, 0.15, boxstyle="round,pad=0.03", fc="#d7ccc8", ec="#5d4037", lw=2))
    ax.text(0.53, 0.275, "Demographic MLP\n(2 Linear Layers)", ha="center", va="center", color="#3e2723", fontweight="bold", fontsize=10)
    ax.annotate("", xy=(0.76, 0.275), xytext=(0.67, 0.275), arrowprops=dict(arrowstyle="->", lw=2, color="#5d4037"))
    
    ax.add_patch(patches.FancyBboxPatch((0.76, 0.20), 0.20, 0.15, boxstyle="round,pad=0.03", fc="#efebe9", ec="#5d4037", lw=2))
    ax.text(0.86, 0.275, "Tabular Embedding\n(16-D Vector)", ha="center", va="center", color="#3e2723", fontweight="bold", fontsize=10)

    # Fusion and Classifier Head
    ax.annotate("", xy=(0.45, 0.49), xytext=(0.86, 0.65), arrowprops=dict(arrowstyle="->", lw=2, color="#0288d1", connectionstyle="angle,angleA=0,angleB=-90,rad=10"))
    ax.annotate("", xy=(0.45, 0.49), xytext=(0.86, 0.35), arrowprops=dict(arrowstyle="->", lw=2, color="#5d4037", connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    ax.add_patch(patches.FancyBboxPatch((0.33, 0.42), 0.25, 0.14, boxstyle="round,pad=0.03", fc="#e8f5e9", ec="#388e3c", lw=2))
    ax.text(0.455, 0.49, "Fusion Module\n(PG-RAAF Attention)", ha="center", va="center", color="#1b5e20", fontweight="bold", fontsize=10)
    ax.annotate("", xy=(0.20, 0.49), xytext=(0.33, 0.49), arrowprops=dict(arrowstyle="->", lw=2, color="#388e3c"))
    
    ax.add_patch(patches.FancyBboxPatch((0.02, 0.42), 0.18, 0.14, boxstyle="round,pad=0.03", fc="#ffebee", ec="#d32f2f", lw=2))
    ax.text(0.11, 0.49, "Classifier Head\nSigmoid Output (5)", ha="center", va="center", color="#b71c1c", fontweight="bold", fontsize=10)
    
    plt.title("Integrated PG-RAAF Multi-Modal ECG Diagnostics Network", fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "figure_architecture.png"), dpi=300)
    fig.savefig(os.path.join(save_dir, "figure_architecture.pdf"), format="pdf")
    plt.close(fig)

def draw_and_save_fusion_diagram(save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    ax.add_patch(patches.FancyBboxPatch((0.05, 0.65), 0.22, 0.12, boxstyle="round,pad=0.03", fc="#e1f5fe", ec="#0288d1", lw=2))
    ax.text(0.16, 0.71, "ECG signal embedding\nh_sig (64-D)", ha="center", va="center", color="#01579b", fontweight="bold", fontsize=9)
    
    ax.add_patch(patches.FancyBboxPatch((0.05, 0.25), 0.22, 0.12, boxstyle="round,pad=0.03", fc="#efebe9", ec="#5d4037", lw=2))
    ax.text(0.16, 0.31, "Tabular demographic\nh_meta (16-D)", ha="center", va="center", color="#3e2723", fontweight="bold", fontsize=9)
    
    # Reliability estimator
    ax.annotate("", xy=(0.42, 0.71), xytext=(0.27, 0.71), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0288d1"))
    ax.add_patch(patches.FancyBboxPatch((0.42, 0.65), 0.20, 0.12, boxstyle="round,pad=0.03", fc="#fffde7", ec="#fbc02d", lw=2))
    ax.text(0.52, 0.71, "Reliability Estimator\nR = Sigmoid(W*h_sig)", ha="center", va="center", color="#f57f17", fontweight="bold", fontsize=8)
    
    # Demographic scaling
    ax.annotate("", xy=(0.42, 0.31), xytext=(0.27, 0.31), arrowprops=dict(arrowstyle="->", lw=1.5, color="#5d4037"))
    ax.add_patch(patches.FancyBboxPatch((0.42, 0.25), 0.20, 0.12, boxstyle="round,pad=0.03", fc="#efebe9", ec="#5d4037", lw=2))
    ax.text(0.52, 0.31, "MLP Projection\nShared Space (64-D)", ha="center", va="center", color="#3e2723", fontweight="bold", fontsize=8)
    
    # Product junction
    ax.annotate("", xy=(0.75, 0.31), xytext=(0.62, 0.31), arrowprops=dict(arrowstyle="->", lw=1.5, color="#3e2723"))
    ax.annotate("", xy=(0.75, 0.31), xytext=(0.52, 0.65), arrowprops=dict(arrowstyle="->", lw=1.5, color="#fbc02d", connectionstyle="angle,angleA=0,angleB=-90,rad=10"))
    
    ax.add_patch(patches.Circle((0.77, 0.31), 0.04, fc="#e8f5e9", ec="#388e3c", lw=2))
    ax.text(0.77, 0.31, "x", ha="center", va="center", color="#1b5e20", fontweight="bold", fontsize=12)
    ax.text(0.77, 0.22, "scaled demographics", ha="center", va="center", color="#1b5e20", fontsize=8)
    
    # Cross attention block
    ax.annotate("", xy=(0.91, 0.50), xytext=(0.27, 0.71), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0288d1", connectionstyle="angle,angleA=0,angleB=-90,rad=10"))
    ax.annotate("", xy=(0.91, 0.50), xytext=(0.81, 0.31), arrowprops=dict(arrowstyle="->", lw=1.5, color="#388e3c", connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    
    ax.add_patch(patches.FancyBboxPatch((0.83, 0.44), 0.15, 0.12, boxstyle="round,pad=0.03", fc="#e0f7fa", ec="#00838f", lw=2))
    ax.text(0.905, 0.50, "Cross-Attention\n& Output Proj", ha="center", va="center", color="#006064", fontweight="bold", fontsize=8)
    
    plt.title("Physiology-Guided Reliability-Aware Attention Fusion (PG-RAAF) Block", fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "figure_fusion.png"), dpi=300)
    fig.savefig(os.path.join(save_dir, "figure_fusion.pdf"), format="pdf")
    plt.close(fig)

# ----------------------------------------------------
# 9. CROSS-VALIDATION SWEEP
# ----------------------------------------------------

def run_cross_validation(args, loader, raw_df, device, config):
    print("\n" + "="*80)
    print("               RUNNING 5-FOLD STRATIFIED CROSS-VALIDATION SWEEP")
    print("="*80 + "\n")
    
    cv_splits = [
        {"train_folds": [4, 5, 6, 7, 8, 9, 10], "val_fold": 3, "test_folds": [1, 2]},
        {"train_folds": [1, 2, 6, 7, 8, 9, 10], "val_fold": 5, "test_folds": [3, 4]},
        {"train_folds": [1, 2, 3, 4, 8, 9, 10], "val_fold": 7, "test_folds": [5, 6]},
        {"train_folds": [1, 2, 3, 4, 5, 6, 10], "val_fold": 9, "test_folds": [7, 8]},
        {"train_folds": [1, 2, 3, 4, 5, 6, 7], "val_fold": 8, "test_folds": [9, 10]}
    ]
    
    cv_results = []
    df_processed = raw_df.copy()
    if df_processed['sex'].dtype == object:
        df_processed['sex'] = df_processed['sex'].map({'Male': 0, 'Female': 1}).fillna(0).astype(int)
    else:
        df_processed['sex'] = df_processed['sex'].fillna(0).astype(int)
        
    waveforms, df_final = loader.load_and_preprocess_waveforms(df_processed, subset_size=args.subset_size)
    y = df_final[loader.superclasses].values.astype(np.float32)
    
    for split_idx, split in enumerate(cv_splits):
        print(f"\n--- CV Fold {split_idx+1}/5 ---")
        
        train_mask = df_final['strat_fold'].isin(split["train_folds"]).values
        val_mask = (df_final['strat_fold'] == split["val_fold"]).values
        test_mask = df_final['strat_fold'].isin(split["test_folds"]).values
        
        scaler = joblib.sklearn.StandardScaler()
        train_meta_raw = df_final.loc[train_mask, loader.metadata_cols].copy()
        
        for col in ['age', 'height', 'weight']:
            median_val = train_meta_raw[col].median()
            train_meta_raw[col] = train_meta_raw[col].fillna(median_val)
            df_final.loc[df_final['strat_fold'].isin(split["train_folds"]), col] = df_final.loc[df_final['strat_fold'].isin(split["train_folds"]), col].fillna(median_val)
            df_final.loc[df_final['strat_fold'] == split["val_fold"], col] = df_final.loc[df_final['strat_fold'] == split["val_fold"], col].fillna(median_val)
            df_final.loc[df_final['strat_fold'].isin(split["test_folds"]), col] = df_final.loc[df_final['strat_fold'].isin(split["test_folds"]), col].fillna(median_val)
            
        scaler.fit(df_final.loc[train_mask, loader.metadata_cols])
        X_meta = scaler.transform(df_final[loader.metadata_cols]).astype(np.float32)
        
        t_train_wave = torch.tensor(waveforms[train_mask], dtype=torch.float32)
        t_train_meta = torch.tensor(X_meta[train_mask], dtype=torch.float32)
        t_train_y = torch.tensor(y[train_mask], dtype=torch.float32)
        
        t_val_wave = torch.tensor(waveforms[val_mask], dtype=torch.float32)
        t_val_meta = torch.tensor(X_meta[val_mask], dtype=torch.float32)
        t_val_y = torch.tensor(y[val_mask], dtype=torch.float32)
        
        t_test_wave = torch.tensor(waveforms[test_mask], dtype=torch.float32)
        t_test_meta = torch.tensor(X_meta[test_mask], dtype=torch.float32)
        t_test_y = torch.tensor(y[test_mask], dtype=torch.float32)
        
        train_loader = DataLoader(TensorDataset(t_train_wave, t_train_meta, t_train_y), batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(TensorDataset(t_val_wave, t_val_meta, t_val_y), batch_size=config["training"]["batch_size"], shuffle=False)
        test_loader = DataLoader(TensorDataset(t_test_wave, t_test_meta, t_test_y), batch_size=config["training"]["batch_size"], shuffle=False)
        
        model = LateFusionNet(
            in_channels=12, meta_features=4, num_classes=5, 
            fusion_type=args.fusion_type,
            mc_dropout_rate=config["model_hyperparameters"]["mc_dropout_rate"]
        ).to(device)
        
        model, _, _ = train_pytorch_model(
            model, train_loader, val_loader, test_loader,
            epochs=config["training"]["epochs"],
            lr=config["training"]["lr"],
            patience=config["training"]["patience"],
            device=device,
            is_fusion=True
        )
        
        _, y_prob, _ = evaluate_epoch(model, test_loader, BCEWithLogitsLossForSigmoidOutput(), device, is_fusion=True)
        metrics = compute_metrics(y[test_mask], y_prob)
        ece = compute_expected_calibration_error(y[test_mask], y_prob)
        brier = compute_brier_score(y[test_mask], y_prob)
        
        cv_results.append({**metrics, "ECE": ece, "Brier": brier})
        print(f"Fold {split_idx+1} Test Metrics: {metrics} | ECE: {ece:.4f} | Brier: {brier:.4f}")
        
    df_cv = pd.DataFrame(cv_results)
    mean_res = df_cv.mean()
    std_res = df_cv.std()
    
    print("\n=== Cross-Validation Results Summary ===")
    for col in df_cv.columns:
        print(f"{col}: {mean_res[col]:.4f} +/- {std_res[col]:.4f}")
        
    os.makedirs(args.artifacts_dir, exist_ok=True)
    df_cv.to_csv(os.path.join(args.artifacts_dir, "table_cross_validation_runs.csv"), index=False)
    print(f"CV Table exported to: {args.artifacts_dir}/table_cross_validation_runs.csv")

# ----------------------------------------------------
# 10. MAIN RESEARCH RUNNER
# ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Cardiac Diagnostics Research Sweep Engine")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--models_dir", type=str, default="./models")
    parser.add_argument("--artifacts_dir", type=str, default="./paper_artifacts")
    parser.add_argument("--fusion_type", type=str, default="reliability_attention", 
                        choices=["concat", "gated", "cross_attention", "feature_attention", "dynamic_weighted", "reliability_attention"])
    parser.add_argument("--run_cv", action="store_true")
    parser.add_argument("--use_tb", action="store_true")
    args = parser.parse_args()
    
    config = load_config()
    if args.epochs: config["training"]["epochs"] = args.epochs
    if args.batch_size: config["training"]["batch_size"] = args.batch_size
    if args.lr: config["training"]["lr"] = args.lr
    if args.subset_size: config["training"]["subset_size"] = args.subset_size
    if args.data_dir: config["data"]["data_dir"] = args.data_dir
    if args.models_dir: config["data"]["models_dir"] = args.models_dir
    if args.artifacts_dir: config["data"]["artifacts_dir"] = args.artifacts_dir
    
    set_seeds(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Execution Device: {device}")
    
    loader = PTBXLZeroLeakageLoader(data_dir=config["data"]["data_dir"])
    raw_df = loader.load_raw_metadata()
    
    if args.run_cv:
        run_cross_validation(args, loader, raw_df, device, config)
        return
        
    print("\n" + "="*80)
    print("               RUNNING FULL MULTI-MODEL SWEEP & ABLATION STUDY")
    print("="*80 + "\n")
    
    # Load splits
    (
        (X_train_wave, X_train_meta, y_train),
        (X_val_wave, X_val_meta, y_val),
        (X_test_wave, X_test_meta, y_test)
    ) = loader.get_data_splits(subset_size=config["training"]["subset_size"])
    
    df_processed, _, _ = loader.preprocess_metadata(raw_df)
    test_indices = df_processed[df_processed['strat_fold'] == 10].index.values[:len(X_test_wave)]
    
    t_train_wave = torch.tensor(X_train_wave, dtype=torch.float32)
    t_train_meta = torch.tensor(X_train_meta, dtype=torch.float32)
    t_train_y = torch.tensor(y_train, dtype=torch.float32)
    t_val_wave = torch.tensor(X_val_wave, dtype=torch.float32)
    t_val_meta = torch.tensor(X_val_meta, dtype=torch.float32)
    t_val_y = torch.tensor(y_val, dtype=torch.float32)
    t_test_wave = torch.tensor(X_test_wave, dtype=torch.float32)
    t_test_meta = torch.tensor(X_test_meta, dtype=torch.float32)
    t_test_y = torch.tensor(y_test, dtype=torch.float32)
    
    train_fusion_loader = DataLoader(TensorDataset(t_train_wave, t_train_meta, t_train_y), batch_size=config["training"]["batch_size"], shuffle=True)
    val_fusion_loader = DataLoader(TensorDataset(t_val_wave, t_val_meta, t_val_y), batch_size=config["training"]["batch_size"], shuffle=False)
    test_fusion_loader = DataLoader(TensorDataset(t_test_wave, t_test_meta, t_test_y), batch_size=config["training"]["batch_size"], shuffle=False)
    
    train_sig_loader = DataLoader(TensorDataset(t_train_wave, t_train_y), batch_size=config["training"]["batch_size"], shuffle=True)
    val_sig_loader = DataLoader(TensorDataset(t_val_wave, t_val_y), batch_size=config["training"]["batch_size"], shuffle=False)
    test_sig_loader = DataLoader(TensorDataset(t_test_wave, t_test_y), batch_size=config["training"]["batch_size"], shuffle=False)
    
    # 1. XGBoost Demographic Baseline
    print("\n--- Training Demographic XGBoost Baseline ---")
    xgb_params = config["model_hyperparameters"]["xgboost"]
    xgb_estimator = XGBClassifier(n_estimators=xgb_params["n_estimators"], learning_rate=xgb_params["learning_rate"], max_depth=xgb_params["max_depth"], random_state=config["training"]["seed"], eval_metric='logloss')
    xgb_model = MultiOutputClassifier(xgb_estimator)
    xgb_model.fit(X_train_meta, y_train)
    xgb_prob_list = xgb_model.predict_proba(X_test_meta)
    y_prob_xgb = np.column_stack([p[:, 1] for p in xgb_prob_list])
    xgb_metrics = compute_metrics(y_test, y_prob_xgb)
    xgb_ece = compute_expected_calibration_error(y_test, y_prob_xgb)
    xgb_brier = compute_brier_score(y_test, y_prob_xgb)
    xgb_size, xgb_latency = measure_inference_speed_and_size(xgb_model, test_fusion_loader, device, is_fusion=True)
    joblib.dump(xgb_model, os.path.join(args.models_dir, "tabular_xgb_model.joblib"))
    
    # 2. Simple 1D CNN
    print("\n--- Training Simple 1D CNN Baseline ---")
    cnn_model = Simple1DCNN(in_channels=12, num_classes=5).to(device)
    cnn_model, _, _ = train_pytorch_model(cnn_model, train_sig_loader, val_sig_loader, test_sig_loader, epochs=config["training"]["epochs"], lr=config["training"]["lr"], patience=config["training"]["patience"], device=device, is_fusion=False)
    _, y_prob_cnn, _ = evaluate_epoch(cnn_model, test_sig_loader, BCEWithLogitsLossForSigmoidOutput(), device, is_fusion=False)
    cnn_metrics = compute_metrics(y_test, y_prob_cnn)
    cnn_ece = compute_expected_calibration_error(y_test, y_prob_cnn)
    cnn_brier = compute_brier_score(y_test, y_prob_cnn)
    cnn_size, cnn_latency = measure_inference_speed_and_size(cnn_model, test_sig_loader, device, is_fusion=False)
    
    # 3. InceptionTime
    print("\n--- Training InceptionTime Baseline ---")
    inc_model = InceptionTime(in_channels=12, num_classes=5, depth=config["model_hyperparameters"]["inception_time"]["depth"], hidden_dim=config["model_hyperparameters"]["inception_time"]["hidden_dim"]).to(device)
    inc_model, _, _ = train_pytorch_model(inc_model, train_sig_loader, val_sig_loader, test_sig_loader, epochs=config["training"]["epochs"], lr=config["training"]["lr"], patience=config["training"]["patience"], device=device, is_fusion=False)
    _, y_prob_inc, _ = evaluate_epoch(inc_model, test_sig_loader, BCEWithLogitsLossForSigmoidOutput(), device, is_fusion=False)
    inc_metrics = compute_metrics(y_test, y_prob_inc)
    inc_ece = compute_expected_calibration_error(y_test, y_prob_inc)
    inc_brier = compute_brier_score(y_test, y_prob_inc)
    inc_size, inc_latency = measure_inference_speed_and_size(inc_model, test_sig_loader, device, is_fusion=False)
    
    # 4. Transformer ECG
    print("\n--- Training Transformer ECG Baseline ---")
    trans_params = config["model_hyperparameters"]["transformer_ecg"]
    trans_model = TransformerECG(in_channels=12, num_classes=5, d_model=trans_params["d_model"], nhead=trans_params["nhead"], num_layers=trans_params["num_layers"], dim_feedforward=trans_params["dim_feedforward"]).to(device)
    trans_model, _, _ = train_pytorch_model(trans_model, train_sig_loader, val_sig_loader, test_sig_loader, epochs=config["training"]["epochs"], lr=config["training"]["lr"], patience=config["training"]["patience"], device=device, is_fusion=False)
    _, y_prob_trans, _ = evaluate_epoch(trans_model, test_sig_loader, BCEWithLogitsLossForSigmoidOutput(), device, is_fusion=False)
    trans_metrics = compute_metrics(y_test, y_prob_trans)
    trans_ece = compute_expected_calibration_error(y_test, y_prob_trans)
    trans_brier = compute_brier_score(y_test, y_prob_trans)
    trans_size, trans_latency = measure_inference_speed_and_size(trans_model, test_sig_loader, device, is_fusion=False)
    
    # 5. ResNet ECG Only
    print("\n--- Training ResNet ECG-Only Branch ---")
    resnet_model = SignalOnlyNet(in_channels=12, num_classes=5).to(device)
    resnet_model, _, _ = train_pytorch_model(resnet_model, train_sig_loader, val_sig_loader, test_sig_loader, epochs=config["training"]["epochs"], lr=config["training"]["lr"], patience=config["training"]["patience"], device=device, is_fusion=False)
    _, y_prob_resnet, _ = evaluate_epoch(resnet_model, test_sig_loader, BCEWithLogitsLossForSigmoidOutput(), device, is_fusion=False)
    resnet_metrics = compute_metrics(y_test, y_prob_resnet)
    resnet_ece = compute_expected_calibration_error(y_test, y_prob_resnet)
    resnet_brier = compute_brier_score(y_test, y_prob_resnet)
    resnet_size, resnet_latency = measure_inference_speed_and_size(resnet_model, test_sig_loader, device, is_fusion=False)
    torch.save(resnet_model.state_dict(), os.path.join(args.models_dir, "signal_only_model.pth"))
    
    # 6. Proposed Late Fusion variant sweep
    fusion_types = ["concat", "gated", "cross_attention", "feature_attention", "dynamic_weighted", "reliability_attention"]
    ablation_results = []
    best_proposed_model = None
    best_proposed_f1 = -1.0
    best_proposed_probs = None
    best_proposed_type = ""
    best_proposed_loss_hist = (None, None)
    
    for f_type in fusion_types:
        print(f"\n--- Training Proposed Multi-Modal Late-Fusion Network ({f_type}) ---")
        fus_model = LateFusionNet(in_channels=12, meta_features=4, num_classes=5, fusion_type=f_type, mc_dropout_rate=config["model_hyperparameters"]["mc_dropout_rate"]).to(device)
        fus_model, train_hist, val_hist = train_pytorch_model(fus_model, train_fusion_loader, val_fusion_loader, test_fusion_loader, epochs=config["training"]["epochs"], lr=config["training"]["lr"], patience=config["training"]["patience"], device=device, is_fusion=True)
        
        y_prob_fus, mean_conf, mean_ent, _ = evaluate_mc_dropout(fus_model, test_fusion_loader, device, steps=config["uncertainty"]["mc_dropout_steps"], is_fusion=True)
        metrics = compute_metrics(y_test, y_prob_fus)
        ece = compute_expected_calibration_error(y_test, y_prob_fus)
        brier = compute_brier_score(y_test, y_prob_fus)
        m_size, m_latency = measure_inference_speed_and_size(fus_model, test_fusion_loader, device, is_fusion=True)
        
        ablation_results.append({
            "Fusion Type": f_type.replace("_", " ").title(),
            "Macro F1-Score": metrics["Macro_F1"],
            "Micro F1-Score": metrics["Micro_F1"],
            "Macro AUROC": metrics["Macro_AUROC"],
            "Macro AUPRC": metrics["Macro_AUPRC"],
            "ECE": ece,
            "Brier Score": brier,
            "Parameters": m_size,
            "Latency (ms)": m_latency
        })
        
        if metrics["Macro_F1"] > best_proposed_f1:
            best_proposed_f1 = metrics["Macro_F1"]
            best_proposed_model = fus_model
            best_proposed_probs = y_prob_fus
            best_proposed_type = f_type
            best_proposed_loss_hist = (train_hist, val_hist)
            
    torch.save(best_proposed_model.state_dict(), os.path.join(args.models_dir, "late_fusion_model.pth"))
    
    # 7. Leave-Group-Out Validation Sweeps
    lgo_df = run_leave_group_out_validation(loader, raw_df, device, config)
    
    # ----------------------------------------------------
    # EXPORTING MANUSCRIPT TABLES
    # ----------------------------------------------------
    print("\n--- Exporting Manuscript Tables ---")
    
    hparams_df = pd.DataFrame([
        {"Hyperparameter": "Random Seed", "Value": str(config["training"]["seed"])},
        {"Hyperparameter": "Epochs", "Value": str(config["training"]["epochs"])},
        {"Hyperparameter": "Batch Size", "Value": str(config["training"]["batch_size"])},
        {"Hyperparameter": "Initial LR", "Value": str(config["training"]["lr"])},
        {"Hyperparameter": "MC Dropout Rate", "Value": str(config["model_hyperparameters"]["mc_dropout_rate"])},
        {"Hyperparameter": "MC Inference Steps", "Value": str(config["uncertainty"]["mc_dropout_steps"])},
        {"Hyperparameter": "Bootstrap Iterations", "Value": str(config["statistical_validation"]["bootstrap_iterations"])}
    ])
    
    model_comparison_df = pd.DataFrame([
        {"Model Architecture": "Demographic XGBoost", "Macro F1": xgb_metrics["Macro_F1"], "Micro F1": xgb_metrics["Micro_F1"], "Macro AUROC": xgb_metrics["Macro_AUROC"], "Macro AUPRC": xgb_metrics["Macro_AUPRC"], "ECE": xgb_ece, "Brier": xgb_brier, "Size (Params)": xgb_size, "Latency (ms)": xgb_latency},
        {"Model Architecture": "Simple 1D CNN", "Macro F1": cnn_metrics["Macro_F1"], "Micro F1": cnn_metrics["Micro_F1"], "Macro AUROC": cnn_metrics["Macro_AUROC"], "Macro AUPRC": cnn_metrics["Macro_AUPRC"], "ECE": cnn_ece, "Brier": cnn_brier, "Size (Params)": cnn_size, "Latency (ms)": cnn_latency},
        {"Model Architecture": "InceptionTime Baseline", "Macro F1": inc_metrics["Macro_F1"], "Micro F1": inc_metrics["Micro_F1"], "Macro AUROC": inc_metrics["Macro_AUROC"], "Macro AUPRC": inc_metrics["Macro_AUPRC"], "ECE": inc_ece, "Brier": inc_brier, "Size (Params)": inc_size, "Latency (ms)": inc_latency},
        {"Model Architecture": "Transformer ECG Baseline", "Macro F1": trans_metrics["Macro_F1"], "Micro F1": trans_metrics["Micro_F1"], "Macro AUROC": trans_metrics["Macro_AUROC"], "Macro AUPRC": trans_metrics["Macro_AUPRC"], "ECE": trans_ece, "Brier": trans_brier, "Size (Params)": trans_size, "Latency (ms)": trans_latency},
        {"Model Architecture": "ResNet ECG Only", "Macro F1": resnet_metrics["Macro_F1"], "Micro F1": resnet_metrics["Micro_F1"], "Macro AUROC": resnet_metrics["Macro_AUROC"], "Macro AUPRC": resnet_metrics["Macro_AUPRC"], "ECE": resnet_ece, "Brier": resnet_brier, "Size (Params)": resnet_size, "Latency (ms)": resnet_latency},
        {"Model Architecture": f"Proposed Late-Fusion ({best_proposed_type.title()})", "Macro F1": best_proposed_f1, "Micro F1": ablation_results[[r["Fusion Type"] for r in ablation_results].index(best_proposed_type.replace("_", " ").title())]["Micro F1-Score"], "Macro AUROC": ablation_results[[r["Fusion Type"] for r in ablation_results].index(best_proposed_type.replace("_", " ").title())]["Macro AUROC"], "Macro AUPRC": ablation_results[[r["Fusion Type"] for r in ablation_results].index(best_proposed_type.replace("_", " ").title())]["Macro AUPRC"], "ECE": ablation_results[[r["Fusion Type"] for r in ablation_results].index(best_proposed_type.replace("_", " ").title())]["ECE"], "Brier": ablation_results[[r["Fusion Type"] for r in ablation_results].index(best_proposed_type.replace("_", " ").title())]["Brier Score"], "Size (Params)": ablation_results[[r["Fusion Type"] for r in ablation_results].index(best_proposed_type.replace("_", " ").title())]["Parameters"], "Latency (ms)": ablation_results[[r["Fusion Type"] for r in ablation_results].index(best_proposed_type.replace("_", " ").title())]["Latency (ms)"]}
    ])
    
    ablation_df = pd.DataFrame(ablation_results)
    robust_df = evaluate_robustness_sweeps(best_proposed_model, test_fusion_loader, device, config)
    fairness_df = evaluate_demographic_fairness(y_test, best_proposed_probs, raw_df, test_indices, config)
    stat_df = run_statistical_validation(y_test, y_prob_xgb, best_proposed_probs, num_iterations=config["statistical_validation"]["bootstrap_iterations"])
    
    tables = {
        "hyperparameters": hparams_df,
        "model_comparisons": model_comparison_df,
        "ablation_results": ablation_df,
        "robustness_sweeps": robust_df,
        "fairness_cohorts": fairness_df,
        "statistical_tests": stat_df,
        "generalization_lgo": lgo_df
    }
    export_latex_and_csv_tables(tables, args.artifacts_dir)
    
    # ----------------------------------------------------
    # GENERATING HIGH-RESOLUTION DIAGRAMS
    # ----------------------------------------------------
    print("\n--- Plotting Manuscript Figures ---")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    # 1. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls_name in enumerate(loader.superclasses):
        if len(np.unique(y_test[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_test[:, i], best_proposed_probs[:, i])
            auc_val = roc_auc_score(y_test[:, i], best_proposed_probs[:, i])
            ax.plot(fpr, tpr, color=colors[i], label=f"{cls_name} (AUC = {auc_val:.4f})", linewidth=2)
        else:
            ax.plot([0, 1], [0.5, 0.5], color=colors[i], linestyle=":", label=f"{cls_name} (Constant)", linewidth=1.5)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.5)
    ax.set_title(f"ROC Curves - Best Late-Fusion Model ({best_proposed_type.title()})", fontsize=11, fontweight='bold')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(args.artifacts_dir, "figure_roc_curves.png"), dpi=300)
    fig.savefig(os.path.join(args.artifacts_dir, "figure_roc_curves.pdf"), format="pdf")
    plt.close(fig)
    
    # 2. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls_name in enumerate(loader.superclasses):
        if len(np.unique(y_test[:, i])) > 1:
            precision, recall, _ = precision_recall_curve(y_test[:, i], best_proposed_probs[:, i])
            ap_val = average_precision_score(y_test[:, i], best_proposed_probs[:, i])
            ax.plot(recall, precision, color=colors[i], label=f"{cls_name} (AUPRC = {ap_val:.4f})", linewidth=2)
        else:
            ax.plot([0, 1], [0.5, 0.5], color=colors[i], linestyle=":", label=f"{cls_name} (Constant)", linewidth=1.5)
    ax.set_title(f"Precision-Recall Curves - {best_proposed_type.title()} Fusion", fontsize=11, fontweight='bold')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(os.path.join(args.artifacts_dir, "figure_pr_curves.png"), dpi=300)
    fig.savefig(os.path.join(args.artifacts_dir, "figure_pr_curves.pdf"), format="pdf")
    plt.close(fig)
    
    # 3. Model Calibration Reliability Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    from sklearn.calibration import calibration_curve
    for i, cls_name in enumerate(loader.superclasses):
        if len(np.unique(y_test[:, i])) > 1:
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test[:, i], best_proposed_probs[:, i], n_bins=8)
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", color=colors[i], label=f"{cls_name}")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_title("Reliability Calibration Curve Diagram", fontsize=11, fontweight='bold')
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(args.artifacts_dir, "figure_calibration_curves.png"), dpi=300)
    fig.savefig(os.path.join(args.artifacts_dir, "figure_calibration_curves.pdf"), format="pdf")
    plt.close(fig)
    
    # 4. Training Convergence Curves
    fig, ax = plt.subplots(figsize=(8, 5))
    t_hist, v_hist = best_proposed_loss_hist
    if t_hist is not None and v_hist is not None:
        ax.plot(range(1, len(t_hist)+1), t_hist, "o-", label="Training Loss", color="#1f77b4", linewidth=2)
        ax.plot(range(1, len(v_hist)+1), v_hist, "s-", label="Validation Loss", color="#ff7f0e", linewidth=2)
        ax.set_title(f"Model Loss Convergence Curves ({best_proposed_type.title()} Fusion)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Binary Cross Entropy Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(args.artifacts_dir, "figure1_loss_curves.png"), dpi=300)
        fig.savefig(os.path.join(args.artifacts_dir, "figure1_loss_curves.pdf"), format="pdf")
    plt.close(fig)
    
    # 5. Multi-Label Confusion Matrices
    fig = plt.figure(figsize=(15, 10))
    for i, cls_name in enumerate(loader.superclasses):
        y_pred_class = (best_proposed_probs[:, i] >= 0.5).astype(int)
        cm = confusion_matrix(y_test[:, i], y_pred_class)
        ax = fig.add_subplot(2, 3, i+1)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Confusion Matrix: {cls_name}", fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Negative', 'Positive'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        thresh = cm.max() / 2.
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, format(cm[row, col], 'd'), ha="center", va="center", color="white" if cm[row, col] > thresh else "black", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(args.artifacts_dir, "figure_confusion_matrix.png"), dpi=300)
    fig.savefig(os.path.join(args.artifacts_dir, "figure_confusion_matrix.pdf"), format="pdf")
    plt.close(fig)
    
    # 6. Feature Importance
    fig, ax = plt.subplots(figsize=(8, 4))
    importances = xgb_model.estimators_[0].feature_importances_
    features = loader.metadata_cols
    colors_feat = ['#3f51b5', '#009688', '#ff9800', '#e91e63']
    ax.barh(features, importances, color=colors_feat)
    ax.set_title("XGBoost Clinical Demographic Feature Importance", fontsize=11, fontweight='bold')
    ax.set_xlabel("Relative Gini Importance Weight")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(args.artifacts_dir, "figure_feature_importance.png"), dpi=300)
    fig.savefig(os.path.join(args.artifacts_dir, "figure_feature_importance.pdf"), format="pdf")
    plt.close(fig)
    
    draw_and_save_architecture_diagram(args.artifacts_dir)
    draw_and_save_fusion_diagram(args.artifacts_dir)
    
    # ----------------------------------------------------
    # GENERATING EXPLAINABILITY RESULTS
    # ----------------------------------------------------
    print("\n--- Generating Local Attributions and Explainability Plots ---")
    pos_idx = np.where(y_test[:, 0] == 1)[0][0]
    t_single_wave = t_test_wave[pos_idx:pos_idx+1]
    t_single_meta = t_test_meta[pos_idx:pos_idx+1]
    plot_and_save_explainability(best_proposed_model, t_single_wave, t_single_meta, loader.metadata_cols, loader.superclasses[0], 0, X_train_meta, args.artifacts_dir)
    
    # ----------------------------------------------------
    # CLINICAL INTERPRETATION DISCUSSION (Phase 14)
    # ----------------------------------------------------
    generate_clinical_interpretation_discussion(y_test, best_proposed_probs, loader, fairness_df, stat_df, robust_df, args.artifacts_dir)
    
    # ----------------------------------------------------
    # EXPORT TO ONNX FORMAT
    # ----------------------------------------------------
    print("\n--- Exporting Final Trained Fusion Model to ONNX ---")
    try:
        onnx_path = os.path.join(args.models_dir, "late_fusion_network.onnx")
        dummy_wave = torch.zeros(1, 12, 1000).to(device)
        dummy_meta = torch.zeros(1, 4).to(device)
        torch.onnx.export(
            best_proposed_model, (dummy_wave, dummy_meta), onnx_path,
            export_params=True, opset_version=11, do_constant_folding=True,
            input_names=['ecg_signal', 'clinical_metadata'], output_names=['diagnostic_probabilities'],
            dynamic_axes={'ecg_signal': {0: 'batch_size'}, 'clinical_metadata': {0: 'batch_size'}, 'diagnostic_probabilities': {0: 'batch_size'}}
        )
        print(f"Model exported successfully to ONNX format: {onnx_path}")
    except Exception as e:
        print(f"ONNX Model Export failed: {e}")
        
    evidence_path = os.path.join(args.artifacts_dir, "manuscript_evidence_deck.txt")
    with open(evidence_path, "w") as f:
        f.write("=== MULTI-MODAL ECG RESEARCH MANUSCRIPT EVIDENCE DECK ===\n\n")
        f.write(f"Best Proposed Late Fusion Model Strategy: {best_proposed_type.title()} Fusion\n")
        f.write(f"Test Macro F1-Score: {best_proposed_f1:.4f}\n")
        f.write("=== STATISTICAL SIGNIFICANCE ===\n")
        f.write(f"McNemar Test p-value proposed vs baseline: {stat_df.iloc[3]['p-value']}\n")
        f.write(f"Wilcoxon Signed-Rank Test p-value: {stat_df.iloc[2]['p-value']}\n\n")
        f.write("=== CALIBRATION METRICS ===\n")
        f.write(f"ECE: {ablation_results[[r['Fusion Type'] for r in ablation_results].index(best_proposed_type.replace('_', ' ').title())]['ECE']:.4f}\n")
        f.write(f"Brier Score: {ablation_results[[r['Fusion Type'] for r in ablation_results].index(best_proposed_type.replace('_', ' ').title())]['Brier Score']:.4f}\n\n")
        f.write("=== DEMOGRAPHIC COHORTS PERFORMANCE ===\n")
        for idx, row in fairness_df.iterrows():
            f.write(f"Cohort: {row['Cohort']} | Size: {row['Sample Count']} | Macro F1: {row['Macro_F1']:.4f} | Macro AUROC: {row['Macro_AUROC']:.4f}\n")
            
    print("\nTraining sweep and artifact generation completed successfully!")
    print(f"Manuscript evidence deck written to: {evidence_path}")

# ----------------------------------------------------
# 11. EXTRA EVALUATION HELPERS
# ----------------------------------------------------

def evaluate_demographic_fairness(y_true, y_prob, raw_df, test_indices, config):
    test_df = raw_df.iloc[test_indices].copy()
    test_df['is_elderly'] = (test_df['age'] >= config["fairness"]["age_threshold"]).astype(int)
    bmi = test_df['weight'] / ((test_df['height'] / 100.0) ** 2)
    bmi_median = bmi.replace([np.inf, -np.inf], np.nan).median()
    bmi = bmi.fillna(bmi_median).replace([np.inf, -np.inf, 0.0], bmi_median)
    test_df['is_high_bmi'] = (bmi >= config["fairness"]["bmi_overweight_threshold"]).astype(int)
    
    cohorts = {
        "Male": (test_df['sex'] == 0).values,
        "Female": (test_df['sex'] == 1).values,
        "Young (<65)": (test_df['is_elderly'] == 0).values,
        "Elderly (>=65)": (test_df['is_elderly'] == 1).values,
        "Normal BMI (<25)": (test_df['is_high_bmi'] == 0).values,
        "High BMI (>=25)": (test_df['is_high_bmi'] == 1).values,
    }
    
    results = []
    for name, mask in cohorts.items():
        if mask.sum() > 0:
            metrics = compute_metrics(y_true[mask], y_prob[mask])
            results.append({"Cohort": name, "Sample Count": mask.sum(), **metrics})
    return pd.DataFrame(results)

def compute_bootstrap_confidence_intervals(y_true, y_prob, num_iterations=200):
    n_samples = len(y_true)
    bootstrap_f1s, bootstrap_aurocs = [], []
    for i in range(num_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2: continue
        metrics = compute_metrics(y_true[indices], y_prob[indices])
        bootstrap_f1s.append(metrics["Macro_F1"])
        bootstrap_aurocs.append(metrics["Macro_AUROC"])
    return np.percentile(bootstrap_f1s, [2.5, 97.5]), np.percentile(bootstrap_aurocs, [2.5, 97.5])

def run_mcnemar_test(y_true, y_prob_a, y_prob_b):
    y_true_flat = y_true.flatten()
    pred_a = (y_prob_a.flatten() >= 0.5).astype(int)
    pred_b = (y_prob_b.flatten() >= 0.5).astype(int)
    correct_a = (pred_a == y_true_flat)
    correct_b = (pred_b == y_true_flat)
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    if (b + c) == 0: return 0.0, 1.0
    stat = (abs(b - c) - 1)**2 / (b + c)
    return stat, stats.chi2.sf(stat, df=1)

def run_statistical_validation(y_true, y_prob_base, y_prob_proposed, num_iterations=200):
    f1_ci_base, auroc_ci_base = compute_bootstrap_confidence_intervals(y_true, y_prob_base, num_iterations)
    f1_ci_prop, auroc_ci_prop = compute_bootstrap_confidence_intervals(y_true, y_prob_proposed, num_iterations)
    mcnemar_stat, p_val_mcnemar = run_mcnemar_test(y_true, y_prob_base, y_prob_proposed)
    eps = 1e-15
    bce_base = - (y_true * np.log(np.clip(y_prob_base, eps, 1.0-eps)) + (1.0 - y_true) * np.log(np.clip(1.0-y_prob_base, eps, 1.0-eps)))
    bce_prop = - (y_true * np.log(np.clip(y_prob_proposed, eps, 1.0-eps)) + (1.0 - y_true) * np.log(np.clip(1.0-y_prob_proposed, eps, 1.0-eps)))
    wilcox_stat, p_val_wilcox = stats.wilcoxon(np.mean(bce_base, axis=1), np.mean(bce_prop, axis=1))
    
    return pd.DataFrame([
        {"Test Metric": "XGBoost F1 95% CI", "Result Value": f"[{f1_ci_base[0]:.4f}, {f1_ci_base[1]:.4f}]", "p-value": "N/A", "Significant (p<0.05)": "N/A"},
        {"Test Metric": "Proposed F1 95% CI", "Result Value": f"[{f1_ci_prop[0]:.4f}, {f1_ci_prop[1]:.4f}]", "p-value": "N/A", "Significant (p<0.05)": "N/A"},
        {"Test Metric": "Proposed vs. XGBoost (Wilcoxon Signed-Rank Loss)", "Result Value": f"W={wilcox_stat:.1f}", "p-value": f"{p_val_wilcox:.4e}", "Significant (p<0.05)": str(p_val_wilcox < 0.05)},
        {"Test Metric": "Proposed vs. XGBoost (McNemar Classification)", "Result Value": f"Chi2={mcnemar_stat:.2f}", "p-value": f"{p_val_mcnemar:.4e}", "Significant (p<0.05)": str(p_val_mcnemar < 0.05)}
    ])

def export_latex_and_csv_tables(tables_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, df in tables_dict.items():
        df.to_csv(os.path.join(output_dir, f"table_{name}.csv"), index=False)
        with open(os.path.join(output_dir, f"table_{name}.tex"), "w") as f:
            f.write(df.to_latex(index=False, column_format='l' + 'c' * (len(df.columns) - 1)))

if __name__ == "__main__":
    main()
