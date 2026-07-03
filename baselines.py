import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------
# 1D CNN Baseline
# ----------------------------------------------------
class Simple1DCNN(nn.Module):
    """
    Standard non-residual 1D Convolutional Neural Network baseline.
    """
    def __init__(self, in_channels=12, num_classes=5):
        super(Simple1DCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, num_classes)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        features = self.conv(x).squeeze(-1)
        logits = self.fc(features)
        return self.activation(logits)

# ----------------------------------------------------
# InceptionTime Baseline
# ----------------------------------------------------
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock1D, self).__init__()
        # Bottleneck
        self.bottleneck = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Convolutions of different kernel sizes
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, bias=False)
        
        # MaxPool branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        bot = self.bottleneck(x)
        out3 = self.conv3(bot)
        out5 = self.conv5(bot)
        out7 = self.conv7(bot)
        
        pool = self.maxpool(x)
        out_pool = self.conv_pool(pool)
        
        concat = torch.cat([out3, out5, out7, out_pool], dim=1)
        return self.relu(self.bn(concat))

class InceptionTime(nn.Module):
    """
    InceptionTime 1D CNN Architecture for Time-Series Classification.
    """
    def __init__(self, in_channels=12, num_classes=5, depth=3, hidden_dim=32):
        super(InceptionTime, self).__init__()
        
        layers = []
        curr_channels = in_channels
        for i in range(depth):
            layers.append(InceptionBlock1D(curr_channels, hidden_dim))
            curr_channels = hidden_dim * 4
            
        self.inception_chain = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(curr_channels, num_classes)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        features = self.inception_chain(x)
        features = self.gap(features).squeeze(-1)
        logits = self.fc(features)
        return self.activation(logits)

# ----------------------------------------------------
# Transformer ECG Baseline
# ----------------------------------------------------
class TransformerECG(nn.Module):
    """
    Transformer Encoder Architecture for ECG Multi-lead Time-series processing.
    Input: (B, 12, 1000)
    It projects spatial channels and encodes temporal position along sequence length.
    """
    def __init__(self, in_channels=12, num_classes=5, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super(TransformerECG, self).__init__()
        # Downsample waveform temporal dimension to save memory/computation
        self.downsample = nn.Conv1d(in_channels, d_model, kernel_size=15, stride=5, padding=5)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        # Input x shape: (B, 12, 1000)
        x = self.downsample(x) # (B, d_model, seq_len)
        x = x.transpose(1, 2)   # (B, seq_len, d_model)
        
        # Pass to Transformer Encoder
        out = self.transformer(x) # (B, seq_len, d_model)
        
        # Temporal Average Pooling
        out = torch.mean(out, dim=1) # (B, d_model)
        
        logits = self.fc(out)
        return self.activation(logits)
