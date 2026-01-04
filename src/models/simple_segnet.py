"""
Simple Conv1d Encoder + ConvTranspose3d Decoder for 3D Segmentation.
Designed for reliable overfitting on small datasets.
"""
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SimpleEncoderConfig:
    hidden: int = 256
    layers: int = 4
    kernel_size: int = 5
    latent_dim: int = 512
    input_channels: int = 4  # Changed default to 4 (Signal, X, Y, Z)


@dataclass
class SimpleDecoderConfig:
    base_channels: int = 64
    depth: int = 4  # Number of upsampling layers
    grid_shape: Tuple[int, int, int] = (64, 64, 64)


class Conv1dEncoder(nn.Module):
    """
    Encodes 1D signal (B, C, Length) -> latent vector (B, latent_dim)
    """
    def __init__(self, cfg: SimpleEncoderConfig, input_length: int):
        super().__init__()
        self.cfg = cfg
        
        layers = []
        in_ch = cfg.input_channels  # Use config input_channels
        out_ch = cfg.hidden
        current = input_length
        
        for i in range(cfg.layers):

            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=cfg.kernel_size, padding=cfg.kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ])
            in_ch = out_ch
            current = current // 2
        
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(out_ch * current, cfg.latent_dim)
        self.norm = nn.LayerNorm(cfg.latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, Length)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten -> (B, Features)
        x = self.fc(x)
        x = self.norm(x)
        return x  # (B, latent_dim)


class ConvTranspose3dDecoder(nn.Module):
    """
    Decodes latent vector (B, latent_dim) -> 3D volume (B, 1, D, H, W)
    """
    def __init__(self, cfg: SimpleDecoderConfig, latent_dim: int):
        super().__init__()
        self.cfg = cfg
        self.grid_shape = cfg.grid_shape
        
        # Calculate initial spatial size after fc projection
        # We'll start from 4x4x4 and upsample to 64x64x64
        self.init_size = 4
        init_channels = cfg.base_channels * (2 ** (cfg.depth - 1))  # e.g., 64 * 8 = 512
        
        self.fc = nn.Linear(latent_dim, init_channels * self.init_size ** 3)
        self.init_channels = init_channels
        
        # Upsampling layers: 4 -> 8 -> 16 -> 32 -> 64
        layers = []
        in_ch = init_channels
        for i in range(cfg.depth):
            out_ch = in_ch // 2 if i < cfg.depth - 1 else cfg.base_channels
            layers.extend([
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        
        self.deconv = nn.Sequential(*layers)
        
        # Final output layer
        self.out_conv = nn.Conv3d(cfg.base_channels, 1, kernel_size=3, padding=1)
        
        # Bias initialization trick: assume background is dominant
        # pi = 0.01 -> bias = -log((1-pi)/pi) = -4.6
        nn.init.constant_(self.out_conv.bias, -5.0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim)
        B = z.size(0)
        x = self.fc(z)
        x = x.view(B, self.init_channels, self.init_size, self.init_size, self.init_size)
        x = self.deconv(x)
        x = self.out_conv(x)
        return x  # (B, 1, D, H, W)


class SimpleSegNet(nn.Module):
    """
    Simple segmentation network: Conv1d Encoder + ConvTranspose3d Decoder.
    Takes signal (B, steps, samples) and outputs mask logits (B, 1, D, H, W).
    """
    def __init__(
        self,
        encoder_cfg: SimpleEncoderConfig = None,
        decoder_cfg: SimpleDecoderConfig = None,
        input_length: int = None,
    ):
        super().__init__()
        self.encoder_cfg = encoder_cfg or SimpleEncoderConfig()
        self.decoder_cfg = decoder_cfg or SimpleDecoderConfig()
        
        self.encoder = None  # Built lazily
        self.decoder = ConvTranspose3dDecoder(self.decoder_cfg, self.encoder_cfg.latent_dim)
    
    def build(self, input_length: int) -> None:
        """Build encoder based on input length."""
        self.encoder = Conv1dEncoder(self.encoder_cfg, input_length)
    
    def forward(self, inputs: torch.Tensor) -> dict:
        """
        Args:
            inputs: (B, C, Length) - C=4 (Signal, X, Y, Z) usually
        Returns:
            dict with 'mask_logits': (B, D, H, W)
        """
        if self.encoder is None:
            # Infer input length from tensor shape
            input_length = inputs.shape[-1]
            self.build(input_length)
            self.encoder = self.encoder.to(inputs.device)
        
        # Encode signal to latent
        z = self.encoder(inputs)  # (B, latent_dim)
        
        # Decode to 3D volume
        mask_logits = self.decoder(z)  # (B, 1, D, H, W)
        mask_logits = mask_logits.squeeze(1)  # (B, D, H, W)
        
        return {"mask_logits": mask_logits}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = SimpleSegNet()
    
    # Simulate input: (B, steps, samples) = (4, 64, 256)
    x = torch.randn(4, 64, 256)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output mask_logits shape: {output['mask_logits'].shape}")
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Check output range
    logits = output["mask_logits"]
    probs = torch.sigmoid(logits)
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"Probs range: [{probs.min():.2f}, {probs.max():.2f}]")
    print(f"Probs mean: {probs.mean():.4f}")
