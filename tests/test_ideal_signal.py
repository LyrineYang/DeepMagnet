#!/usr/bin/env python
"""
Direct Mask-Encoded Signal Test.
If the network can learn when signal DIRECTLY encodes mask, 
but fails when using physics signal, then physics signal is the bottleneck.

This test creates synthetic signals that directly encode mask projection.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import VolumeDataset
from src.models.heatmap_segnet import HeatmapSegNet, HeatmapEncoderConfig, VolumeDecoderConfig
from src.models.losses import dice_loss
from src.utils.config import load_yaml


def encode_mask_to_signal(mask):
    """
    Create an 'ideal' signal that perfectly encodes the mask.
    Signal = sum along Z axis (depth projection).
    """
    # mask: (B, 64, 64, 64)
    B = mask.size(0)
    
    # Z-axis projection: sum or max along depth
    projection = mask.max(dim=1)[0]  # (B, 64, 64) - max projection along Z
    
    # Downsample to 32x32
    projection = F.interpolate(projection.unsqueeze(1), size=(32, 32), mode='bilinear', align_corners=False)
    projection = projection.squeeze(1)  # (B, 32, 32)
    
    # Add position encoding
    x_pos = torch.linspace(0, 1, 32, device=mask.device).view(1, 1, 32).expand(B, 32, 32)
    y_pos = torch.linspace(0, 1, 32, device=mask.device).view(1, 32, 1).expand(B, 32, 32)
    
    # Stack as 3 channels
    heatmap = torch.stack([projection, x_pos, y_pos], dim=1)  # (B, 3, 32, 32)
    
    return heatmap


def main():
    print("=" * 60)
    print("DIRECT MASK ENCODING TEST")
    print("Signal = Mask Projection (ideal case)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load dataset
    print("\n1. Loading dataset...")
    data_cfg = load_yaml("configs/data_overfit.yaml")
    train_dir = Path(data_cfg["dataset"]["output_dir"]) / "train"
    
    shard_size = data_cfg["dataset"].get("shard_size")
    train_ds = VolumeDataset(str(train_dir), shard_size=shard_size)
    print(f"   Samples: {len(train_ds)}")
    
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    
    # Build model
    print("\n2. Building HeatmapSegNet...")
    encoder_cfg = HeatmapEncoderConfig(base_channels=64, depth=4)
    decoder_cfg = VolumeDecoderConfig(base_channels=64, depth=5, output_size=(64, 64, 64))
    model = HeatmapSegNet(encoder_cfg, decoder_cfg)
    model = model.to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    pos_weight = torch.tensor([30.0]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    
    # Training
    print("\n3. Training for 50 epochs with IDEAL signal (mask projection)...")
    print("-" * 60)
    
    best_dice = 0.0
    
    for epoch in range(50):
        model.train()
        epoch_dice = 0.0
        n_batches = 0
        
        for batch in loader:
            mask = batch["mask"].to(device)
            
            # Create ideal signal from mask
            heatmap = encode_mask_to_signal(mask)
            
            optimizer.zero_grad()
            
            outputs = model(heatmap)
            logits = outputs["mask_logits"]
            
            bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
            dice = dice_loss(logits, mask)
            loss = 0.5 * bce + 0.5 * dice
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
                intersection = (pred * mask).sum()
                union = pred.sum() + mask.sum()
                dice_score = (2 * intersection / (union + 1e-6)).item()
            
            epoch_dice += dice_score
            n_batches += 1
        
        scheduler.step()
        avg_dice = epoch_dice / n_batches
        
        if avg_dice > best_dice:
            best_dice = avg_dice
        
        if epoch % 10 == 0 or epoch == 49:
            print(f"Epoch {epoch:2d}: dice={avg_dice:.4f}, best={best_dice:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Best Dice with IDEAL signal: {best_dice:.4f}")
    print()
    
    if best_dice > 0.9:
        print("✅ Network can PERFECTLY learn from mask projections!")
        print("   This proves: architecture is fine, physics signal is the problem.")
    elif best_dice > 0.7:
        print("⚠️  Network learns well from mask projections.")
        print("   Physics signal lacks necessary information.")
    else:
        print("❌ Even ideal signal fails - architecture issue.")


if __name__ == "__main__":
    main()
