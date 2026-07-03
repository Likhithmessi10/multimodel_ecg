import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class GradCAM1D:
    """
    Custom 1D Grad-CAM implementation for ECG Convolutional Layers.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_a = target_layer.register_forward_hook(self.save_activation)
        self.hook_g = target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def __call__(self, x_wave, x_meta, target_class_idx, is_fusion=True):
        self.model.zero_grad()
        
        # Forward pass
        if is_fusion:
            probs = self.model(x_wave, x_meta)
        else:
            probs = self.model(x_wave)
            
        # Target probability
        score = probs[0, target_class_idx]
        score.backward()
        
        # Calculate Grad-CAM
        # Pool the gradients across the temporal dimension
        weights = torch.mean(self.gradients, dim=-1, keepdim=True) # (1, channels, 1)
        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze(0) # (time,)
        
        # Apply ReLU
        grad_cam = torch.clamp(grad_cam, min=0.0)
        
        # Normalize
        if grad_cam.max() > 0:
            grad_cam = grad_cam / grad_cam.max()
            
        return grad_cam.cpu().numpy()
        
    def remove_hooks(self):
        self.hook_a.remove()
        self.hook_g.remove()

def compute_integrated_gradients(model, x_wave, x_meta, target_class_idx, steps=50, is_fusion=True):
    """
    Custom 1D Integrated Gradients implementation for ECG signals.
    """
    model.eval()
    x_wave_ref = torch.zeros_like(x_wave) # Baseline (flatline)
    
    # Generate scaled inputs along the path
    scaled_inputs = [x_wave_ref + (float(i) / steps) * (x_wave - x_wave_ref) for i in range(steps + 1)]
    
    grads = []
    for wave_step in scaled_inputs:
        wave_step = wave_step.clone().detach().requires_grad_(True)
        if is_fusion:
            probs = model(wave_step, x_meta)
        else:
            probs = model(wave_step)
            
        score = probs[0, target_class_idx]
        model.zero_grad()
        score.backward()
        grads.append(wave_step.grad.detach().cpu().numpy())
        
    # Average gradients and multiply by path delta
    avg_grads = np.mean(grads, axis=0) # (1, 12, 1000)
    delta = (x_wave - x_wave_ref).cpu().numpy()
    ig = delta * avg_grads
    return ig[0] # (12, 1000)

def compute_exact_shapley_values(model, x_wave, x_meta_row, target_class_idx, base_meta_dataset):
    """
    Computes exact Shapley values for 4 clinical metadata features
    (Age, Sex, Height, Weight) using background dataset expectation.
    """
    model.eval()
    d = len(x_meta_row) # Should be 4
    shapley_values = np.zeros(d)
    
    # Calculate background expectation
    background_mean = np.mean(base_meta_dataset, axis=0)
    
    # Evaluate a coalition vector
    def evaluate_coalition(coalition_mask):
        # Substitute background features where mask is 0
        input_meta = x_meta_row.copy()
        for idx in range(d):
            if coalition_mask[idx] == 0:
                input_meta[idx] = background_mean[idx]
                
        # Run inference
        t_meta = torch.tensor([input_meta], dtype=torch.float32)
        with torch.no_grad():
            probs = model(x_wave, t_meta)
        return probs[0, target_class_idx].item()
        
    # Standard combinatorial weights formula for Shapley value
    from itertools import combinations
    for i in range(d):
        marginal_contributions = []
        features_excluding_i = [idx for idx in range(d) if idx != i]
        
        # Iterate over all possible subsets of other features
        for size in range(d):
            for subset in combinations(features_excluding_i, size):
                # Coalition without feature i
                mask_without = np.zeros(d)
                for idx in subset:
                    mask_without[idx] = 1
                    
                # Coalition with feature i
                mask_with = mask_without.copy()
                mask_with[i] = 1
                
                v_with = evaluate_coalition(mask_with)
                v_without = evaluate_coalition(mask_without)
                
                # Weight = |S|! * (N - |S| - 1)! / N!
                weight = (math.factorial(size) * math.factorial(d - size - 1)) / math.factorial(d)
                marginal_contributions.append(weight * (v_with - v_without))
                
        shapley_values[i] = np.sum(marginal_contributions)
        
    return shapley_values

def plot_and_save_explainability(model, x_wave, x_meta, metadata_cols, target_class_name, target_class_idx, base_meta_dataset, save_dir):
    """
    Generates and saves explainability maps: Grad-CAM heatmap overlay,
    Integrated Gradients attribution per lead, and SHAP feature attributions.
    Saves in both PNG and PDF formats.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Grad-CAM
    target_layer = model.signal_branch.prep[0]
    grad_cam_extractor = GradCAM1D(model, target_layer)
    cam_heatmap = grad_cam_extractor(x_wave, x_meta, target_class_idx, is_fusion=True)
    grad_cam_extractor.remove_hooks()
    
    # Save Grad-CAM Heatmap Plot
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    time_axis = np.arange(1000) / 100.0 # 100 Hz sampling rate
    lead_ii_signal = x_wave[0, 1].cpu().numpy()
    ax1.plot(time_axis, lead_ii_signal, color='black', label='ECG Lead II', alpha=0.8)
    
    # Overlay heatmap
    im1 = ax1.imshow(np.expand_dims(cam_heatmap, 0), aspect='auto', cmap='Reds', alpha=0.4, 
               extent=[time_axis[0], time_axis[-1], lead_ii_signal.min(), lead_ii_signal.max()])
    plt.colorbar(im1, ax=ax1, label='Grad-CAM Importance')
    ax1.set_title(f"Grad-CAM Temporal Localization: Lead II Overlay ({target_class_name})", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Normalized Amplitude")
    ax1.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(save_dir, "figure_attention_maps.png"), dpi=300)
    fig1.savefig(os.path.join(save_dir, "figure_attention_maps.pdf"), format="pdf")
    plt.close(fig1)
    
    # 2. Integrated Gradients
    ig_attrib = compute_integrated_gradients(model, x_wave, x_meta, target_class_idx, is_fusion=True)
    
    # Save Integrated Gradients attribution per lead
    fig2 = plt.figure(figsize=(12, 10))
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for idx, name in enumerate(lead_names):
        ax = fig2.add_subplot(6, 2, idx+1)
        signal = x_wave[0, idx].cpu().numpy()
        attrib = ig_attrib[idx]
        
        # Color coding for negative and positive contributions
        pos_attrib = np.maximum(attrib, 0)
        neg_attrib = np.minimum(attrib, 0)
        
        ax.plot(time_axis, signal, color='grey', alpha=0.5)
        ax.fill_between(time_axis, signal, signal + pos_attrib * 2.0, color='red', alpha=0.7, label='Positive attribution')
        ax.fill_between(time_axis, signal, signal + neg_attrib * 2.0, color='blue', alpha=0.7, label='Negative attribution')
        ax.set_title(f"Lead {name}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', prop={'size': 8})
            
    fig2.suptitle(f"Integrated Gradients Lead-Wise Attribution maps ({target_class_name})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(os.path.join(save_dir, "figure_lead_contribution.png"), dpi=300)
    fig2.savefig(os.path.join(save_dir, "figure_lead_contribution.pdf"), format="pdf")
    plt.close(fig2)
    
    # 3. Exact SHAP values
    x_meta_row = x_meta[0].cpu().numpy()
    shap_vals = compute_exact_shapley_values(model, x_wave, x_meta_row, target_class_idx, base_meta_dataset)
    
    # Save SHAP feature attribution plot
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    colors = ['#FF4B4B' if v >= 0 else '#1f77b4' for v in shap_vals]
    ax3.barh(metadata_cols, shap_vals, color=colors)
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1.0)
    ax3.set_title(f"Exact Shapley Metadata Feature Attributions ({target_class_name})", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Shapley Value Contribution")
    ax3.set_ylabel("Clinical Attribute")
    ax3.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig3.savefig(os.path.join(save_dir, "figure_shap_plots.png"), dpi=300)
    fig3.savefig(os.path.join(save_dir, "figure_shap_plots.pdf"), format="pdf")
    plt.close(fig3)
    
    print("Explainability figures generated successfully in PNG and PDF formats!")

