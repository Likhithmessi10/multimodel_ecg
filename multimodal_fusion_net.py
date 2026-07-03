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
        # Initial convolution to project raw signals to channels
        self.prep = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Block 1: 32 -> 32 channels
        self.block1 = ResBlock1D(32, 32)
        # Block 2: 32 -> 64 channels
        self.block2 = ResBlock1D(32, 64)
        # Block 3: 64 -> 64 channels
        self.block3 = ResBlock1D(64, 64)
        
        # Global Average Pooling along temporal dimension
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: (B, 12, 1000)
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

class SignalOnlyNet(nn.Module):
    """
    Signal-Only Classifier baseline using 1D-ResNet and a linear classification head.
    """
    def __init__(self, in_channels=12, num_classes=5):
        super(SignalOnlyNet, self).__init__()
        self.signal_branch = SignalBranch(in_channels=in_channels)
        self.fc = nn.Linear(64, num_classes)
        self.activation = nn.Sigmoid()
        
    def forward(self, x_wave):
        emb = self.signal_branch(x_wave)
        logits = self.fc(emb)
        return self.activation(logits)

class LateFusionNet(nn.Module):
    """
    Proposed Multi-Modal Late-Fusion Network combining 1D-ResNet and MLP metadata branches.
    """
    def __init__(self, in_channels=12, meta_features=4, num_classes=5):
        super(LateFusionNet, self).__init__()
        self.signal_branch = SignalBranch(in_channels=in_channels)
        self.meta_branch = MetadataBranch(in_features=meta_features)
        
        # Concat late fusion classifier
        # 64 (signal) + 16 (meta) = 80 features
        self.fc = nn.Linear(64 + 16, num_classes)
        self.activation = nn.Sigmoid()
        
    def forward(self, x_wave, x_meta):
        # Extract embeddings
        sig_emb = self.signal_branch(x_wave)  # (B, 64)
        meta_emb = self.meta_branch(x_meta)   # (B, 16)
        
        # Concatenate late fusion
        fused = torch.cat((sig_emb, meta_emb), dim=1) # (B, 80)
        
        # Project onto output logit nodes
        logits = self.fc(fused) # (B, 5)
        return self.activation(logits)
