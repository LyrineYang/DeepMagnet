"""
2D-to-3D Segmentation Network.
Input: (B, 3, 32, 32) - Signal heatmap + XY position encoding
Output: (B, 1, D, H, W) - 3D volume prediction

This design aligns signal pixels with output voxel columns,
making the task similar to depth estimation.
"""
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeatmapEncoderConfig:
    base_channels: int = 64
    depth: int = 4  # 32 -> 16 -> 8 -> 4 -> 2


@dataclass
class VolumeDecoderConfig:
    base_channels: int = 64
    depth: int = 5  # 2 -> 4 -> 8 -> 16 -> 32 -> 64
    output_size: Tuple[int, int, int] = (64, 64, 64)


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class HeatmapEncoder(nn.Module):
    """
    Encodes 2D heatmap (B, 3, 32, 32) -> latent (B, C, 2, 2)
    """
    def __init__(self, cfg: HeatmapEncoderConfig, in_channels: int = 3):
        super().__init__()
        self.cfg = cfg
        
        # Initial conv
        self.init_conv = ConvBlock2D(in_channels, cfg.base_channels)
        
        # Downsampling path
        self.downs = nn.ModuleList()
        ch = cfg.base_channels
        for i in range(cfg.depth):
            out_ch = min(ch * 2, 512)
            self.downs.append(nn.Sequential(
                nn.MaxPool2d(2),
                ConvBlock2D(ch, out_ch),
            ))
            ch = out_ch
        
        self.out_channels = ch
    
    def forward(self, x):
        # x: (B, 3, 32, 32)
        features = [self.init_conv(x)]
        
        for down in self.downs:
            features.append(down(features[-1]))
        
        return features[-1], features  # Final latent + skip connections


class VolumeDecoder(nn.Module):
    """
    Decodes latent (B, C, 2, 2) -> volume (B, 1, 64, 64, 64)
    Uses 2D-to-3D transition via reshape.
    """
    def __init__(self, cfg: VolumeDecoderConfig, latent_channels: int):
        super().__init__()
        self.cfg = cfg
        
        # Expand from 2D (2×2) to 3D (2×2×2)
        self.to_3d = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels * 2, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.latent_channels = latent_channels
        
        # 3D Upsampling path
        self.ups = nn.ModuleList()
        ch = latent_channels
        for i in range(cfg.depth):
            out_ch = max(ch // 2, cfg.base_channels)
            self.ups.append(nn.Sequential(
                nn.ConvTranspose3d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ))
            ch = out_ch
        
        # Final output
        self.out_conv = nn.Conv3d(ch, 1, kernel_size=3, padding=1)
        
        # Bias initialization for class imbalance
        nn.init.constant_(self.out_conv.bias, -5.0)
    
    def forward(self, latent_2d):
        # latent_2d: (B, C, 2, 2)
        B = latent_2d.size(0)
        
        # Expand channels for depth dimension
        x = self.to_3d(latent_2d)  # (B, C*2, 2, 2)
        
        # Reshape to 3D: (B, C, 2, 2, 2)
        x = x.view(B, self.latent_channels, 2, 2, 2)
        
        # Upsample to output size
        for up in self.ups:
            x = up(x)
        
        # Final conv
        x = self.out_conv(x)
        return x  # (B, 1, D, H, W)


class HeatmapSegNet(nn.Module):
    """
    2D Heatmap -> 3D Volume Segmentation.
    Input: (B, 3, H, W) - Signal heatmap with position encoding
    Output: (B, D, H, W) mask logits
    """
    def __init__(
        self,
        encoder_cfg: HeatmapEncoderConfig = None,
        decoder_cfg: VolumeDecoderConfig = None,
    ):
        super().__init__()
        self.encoder_cfg = encoder_cfg or HeatmapEncoderConfig()
        self.decoder_cfg = decoder_cfg or VolumeDecoderConfig()
        
        self.encoder = HeatmapEncoder(self.encoder_cfg, in_channels=3)
        self.decoder = VolumeDecoder(self.decoder_cfg, self.encoder.out_channels)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - 3 channels: signal, x_pos, y_pos
        Returns:
            dict with 'mask_logits': (B, D, H, W)
        """
        latent, _ = self.encoder(x)
        volume = self.decoder(latent)
        volume = volume.squeeze(1)  # (B, D, H, W)
        
        return {"mask_logits": volume}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = HeatmapSegNet()
    
    # Input: (B, 3, 32, 32) - signal heatmap + position encoding
    x = torch.randn(4, 3, 32, 32)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output mask_logits shape: {output['mask_logits'].shape}")
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Check output stats
    logits = output["mask_logits"]
    probs = torch.sigmoid(logits)
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"Probs mean: {probs.mean():.4f}")
