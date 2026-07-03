import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=15, padding=7):
        super(ResBlock1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Shortcut path to match channels and size (downsampled by MaxPool1d)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        shortcut_out = self.shortcut(x)
        
        # Add residual connection before pooling
        out = out + shortcut_out
        out = self.maxpool(out)
        return out

class SignalBranch(nn.Module):
    """
    1D CNN Signal Branch with 3 Residual Blocks.
    Input: (batch_size, 12, 1000)
    Output: 64-dimensional dense signal embedding
    """
    def __init__(self, in_channels=12):
        super(SignalBranch, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.block1 = ResBlock1D(32, 32)
        self.block2 = ResBlock1D(32, 64)
        self.block3 = ResBlock1D(64, 64)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        x = self.prep(x)       # (B, 32, 1000)
        x = self.block1(x)     # (B, 32, 500)
        x = self.block2(x)     # (B, 64, 250)
        x = self.block3(x)     # (B, 64, 125)
        x = self.gap(x)        # (B, 64, 1)
        x = torch.flatten(x, 1) # (B, 64)
        return x

class MetadataBranch(nn.Module):
    """
    MLP Metadata Branch
    Input: (batch_size, 4) - age, sex, height, weight
    Output: 16-dimensional tabular metadata embedding
    """
    def __init__(self, in_features=4):
        super(MetadataBranch, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
    def forward(self, x):
        return self.mlp(x)

# ----------------------------------------------------
# ADVANCED MULTI-MODAL FUSION MODULES
# ----------------------------------------------------

class GatedFusion(nn.Module):
    """
    Gated Fusion Module that learns to control information flow
    from signal and metadata branches dynamically.
    """
    def __init__(self, d_sig=64, d_meta=16, d_out=64):
        super(GatedFusion, self).__init__()
        self.linear_sig = nn.Linear(d_sig, d_out)
        self.linear_meta = nn.Linear(d_meta, d_out)
        self.gate = nn.Linear(d_sig + d_meta, d_out)
        
    def forward(self, h_sig, h_meta):
        # Learn a dynamic gating coefficient vector
        concat = torch.cat([h_sig, h_meta], dim=-1)
        g = torch.sigmoid(self.gate(concat))
        
        # Apply gate to projection of branches
        out = g * self.linear_sig(h_sig) + (1 - g) * self.linear_meta(h_meta)
        return out

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module that dynamically aligns and attends
    signal embeddings based on metadata projections.
    """
    def __init__(self, d_sig=64, d_meta=16, d_model=64):
        super(CrossAttentionFusion, self).__init__()
        self.q_proj = nn.Linear(d_sig, d_model)
        self.k_proj = nn.Linear(d_meta, d_model)
        self.v_proj = nn.Linear(d_meta, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        
    def forward(self, h_sig, h_meta):
        # Shape: (B, 1, d_model)
        q = self.q_proj(h_sig).unsqueeze(1)
        k = self.k_proj(h_meta).unsqueeze(1)
        v = self.v_proj(h_meta).unsqueeze(1)
        
        # Cross-attention weights
        attn_scores = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn_out = (attn_scores @ v).squeeze(1)
        
        return self.out_proj(attn_out) + q.squeeze(1) # Skip connection

class FeatureAttentionFusion(nn.Module):
    """
    Feature Attention Fusion Module that applies self-attention
    across the fused feature space elements.
    """
    def __init__(self, d_in=80):
        super(FeatureAttentionFusion, self).__init__()
        self.query = nn.Linear(d_in, d_in)
        self.key = nn.Linear(d_in, d_in)
        self.value = nn.Linear(d_in, d_in)
        self.scale = d_in ** -0.5
        
    def forward(self, h_fused):
        q = self.query(h_fused)
        k = self.key(h_fused)
        v = self.value(h_fused)
        
        attn = torch.sigmoid((q * k) * self.scale)
        return attn * v + h_fused

class DynamicWeightedFusion(nn.Module):
    """
    Dynamic Weighted Fusion Module that learns scalar weights
    for both signal and metadata embeddings dynamically.
    """
    def __init__(self, d_sig=64, d_meta=16, d_out=64):
        super(DynamicWeightedFusion, self).__init__()
        self.linear_sig = nn.Linear(d_sig, d_out)
        self.linear_meta = nn.Linear(d_meta, d_out)
        self.weight_proj = nn.Linear(d_sig + d_meta, 2)
        
    def forward(self, h_sig, h_meta):
        concat = torch.cat([h_sig, h_meta], dim=-1)
        weights = torch.softmax(self.weight_proj(concat), dim=-1) # (B, 2)
        
        w_sig = weights[:, 0].unsqueeze(-1)
        w_meta = weights[:, 1].unsqueeze(-1)
        
        out = w_sig * self.linear_sig(h_sig) + w_meta * self.linear_meta(h_meta)
        return out

# ----------------------------------------------------
# COMPATIBLE & NOVEL MULTI-MODAL ARCHITECTURES
# ----------------------------------------------------

class SignalOnlyNet(nn.Module):
    """
    Signal-Only Classifier baseline using 1D-ResNet and a linear classification head.
    """
    def __init__(self, in_channels=12, num_classes=5, mc_dropout_rate=0.0):
        super(SignalOnlyNet, self).__init__()
        self.signal_branch = SignalBranch(in_channels=in_channels)
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=mc_dropout_rate) if mc_dropout_rate > 0.0 else nn.Identity()
        self.activation = nn.Sigmoid()
        
    def forward(self, x_wave):
        emb = self.signal_branch(x_wave)
        emb = self.dropout(emb)
        logits = self.fc(emb)
        return self.activation(logits)

class LateFusionNet(nn.Module):
    """
    Multi-Modal Late-Fusion Network combining 1D-ResNet and MLP metadata branches.
    Supports Gated, Cross-Attention, Feature Attention, and Dynamic Weighted Fusion.
    """
    def __init__(self, in_channels=12, meta_features=4, num_classes=5, fusion_type="concat", mc_dropout_rate=0.0):
        super(LateFusionNet, self).__init__()
        self.signal_branch = SignalBranch(in_channels=in_channels)
        self.meta_branch = MetadataBranch(in_features=meta_features)
        self.fusion_type = fusion_type
        self.dropout = nn.Dropout(p=mc_dropout_rate) if mc_dropout_rate > 0.0 else nn.Identity()
        
        # Select Fusion Module
        if fusion_type == "concat":
            self.fusion_dim = 64 + 16 # 80
            self.fusion_module = nn.Identity()
            self.fc = nn.Linear(self.fusion_dim, num_classes)
        elif fusion_type == "gated":
            self.fusion_dim = 64
            self.fusion_module = GatedFusion(d_sig=64, d_meta=16, d_out=64)
            self.fc = nn.Linear(self.fusion_dim, num_classes)
        elif fusion_type == "cross_attention":
            self.fusion_dim = 64
            self.fusion_module = CrossAttentionFusion(d_sig=64, d_meta=16, d_model=64)
            self.fc = nn.Linear(self.fusion_dim, num_classes)
        elif fusion_type == "feature_attention":
            self.fusion_dim = 80
            self.fusion_module = FeatureAttentionFusion(d_in=80)
            self.fc = nn.Linear(self.fusion_dim, num_classes)
        elif fusion_type == "dynamic_weighted":
            self.fusion_dim = 64
            self.fusion_module = DynamicWeightedFusion(d_sig=64, d_meta=16, d_out=64)
            self.fc = nn.Linear(self.fusion_dim, num_classes)
        else:
            raise ValueError(f"Unknown fusion strategy type: {fusion_type}")
            
        self.activation = nn.Sigmoid()
        
    def forward(self, x_wave, x_meta):
        # Extract embeddings
        sig_emb = self.signal_branch(x_wave)  # (B, 64)
        meta_emb = self.meta_branch(x_meta)   # (B, 16)
        
        # Apply fusion
        if self.fusion_type == "concat":
            fused = torch.cat((sig_emb, meta_emb), dim=1) # (B, 80)
            fused = self.fusion_module(fused)
        elif self.fusion_type in ["gated", "cross_attention", "dynamic_weighted"]:
            fused = self.fusion_module(sig_emb, meta_emb) # (B, 64)
        elif self.fusion_type == "feature_attention":
            concat = torch.cat((sig_emb, meta_emb), dim=1) # (B, 80)
            fused = self.fusion_module(concat) # (B, 80)
            
        fused = self.dropout(fused)
        logits = self.fc(fused) # (B, num_classes)
        return self.activation(logits)
