"""
Conv2d Encoder + ConvTranspose3d Decoder for Grid-based Segmentation.
Input: (B, 4, H, W) - 2D grid of signals with position encoding
Output: (B, 1, D, H, W) - 3D volume prediction

Key improvement over Conv1d:
- Conv2d sees BOTH horizontal and vertical neighbors in the grid
- This captures 100% of spatial correlations vs 50% with Conv1d
"""
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Conv2dEncoderConfig:
    base_channels: int = 64
    depth: int = 4
    input_channels: int = 4  # Signal + X + Y + Z
    latent_dim: int = 512


@dataclass
class Decoder3dConfig:
    base_channels: int = 64
    depth: int = 4  # 4 -> 8 -> 16 -> 32 -> 64
    output_size: Tuple[int, int, int] = (64, 64, 64)


class Conv2dBlock(nn.Module):
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


class Conv2dEncoder(nn.Module):
    """
    Encodes 2D grid (B, C, H, W) -> latent vector (B, latent_dim)
    """
    def __init__(self, cfg: Conv2dEncoderConfig, grid_size: int):
        super().__init__()
        self.cfg = cfg
        
        # Initial conv
        self.init_conv = Conv2dBlock(cfg.input_channels, cfg.base_channels)
        
        # Downsampling path
        self.downs = nn.ModuleList()
        ch = cfg.base_channels
        current_size = grid_size
        
        for i in range(cfg.depth):
            out_ch = min(ch * 2, 512)
            self.downs.append(nn.Sequential(
                nn.MaxPool2d(2),
                Conv2dBlock(ch, out_ch),
            ))
            ch = out_ch
            current_size = current_size // 2
        
        self.final_size = current_size
        self.final_channels = ch
        
        # FC to latent
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * current_size * current_size, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.init_conv(x)
        
        for down in self.downs:
            x = down(x)
        
        x = self.fc(x)
        return x  # (B, latent_dim)


class Decoder3d(nn.Module):
    """
    Decodes latent (B, latent_dim) -> volume (B, 1, D, H, W)
    """
    def __init__(self, cfg: Decoder3dConfig, latent_dim: int):
        super().__init__()
        self.cfg = cfg
        
        # Start from 4x4x4
        self.init_size = 4
        init_channels = cfg.base_channels * (2 ** (cfg.depth - 1))
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, init_channels * self.init_size ** 3),
            nn.ReLU(inplace=True),
        )
        self.init_channels = init_channels
        
        # 3D Upsampling: 4 -> 8 -> 16 -> 32 -> 64
        self.ups = nn.ModuleList()
        ch = init_channels
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
        
        # Output layer
        self.out_conv = nn.Conv3d(ch, 1, kernel_size=3, padding=1)
        nn.init.constant_(self.out_conv.bias, -5.0)  # Bias trick
    
    def forward(self, z):
        B = z.size(0)
        x = self.fc(z)
        x = x.view(B, self.init_channels, self.init_size, self.init_size, self.init_size)
        
        for up in self.ups:
            x = up(x)
        
        return self.out_conv(x)  # (B, 1, D, H, W)


class GridSegNet(nn.Module):
    """
    2D Grid -> 3D Volume Segmentation.
    Input: (B, 4, H, W) - 2D grid with 4 channels (Signal, X, Y, Z)
    Output: (B, D, H, W) mask logits
    """
    def __init__(
        self,
        encoder_cfg: Conv2dEncoderConfig = None,
        decoder_cfg: Decoder3dConfig = None,
        grid_size: int = 24,
    ):
        super().__init__()
        self.encoder_cfg = encoder_cfg or Conv2dEncoderConfig()
        self.decoder_cfg = decoder_cfg or Decoder3dConfig()
        self.grid_size = grid_size
        
        self.encoder = Conv2dEncoder(self.encoder_cfg, grid_size)
        self.decoder = Decoder3d(self.decoder_cfg, self.encoder_cfg.latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, 4, H, W) - 2D grid input
        Returns:
            dict with 'mask_logits': (B, D, H, W)
        """
        z = self.encoder(x)
        vol = self.decoder(z)
        vol = vol.squeeze(1)  # (B, D, H, W)
        return {"mask_logits": vol}


if __name__ == "__main__":
    # Test
    model = GridSegNet(grid_size=24)
    x = torch.randn(4, 4, 24, 24)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out['mask_logits'].shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
