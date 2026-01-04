#!/usr/bin/env python
"""
Overfitting Test with Conv2d Encoder + 24x24 Dense Grid.
Key improvements:
1. Conv2d sees both horizontal AND vertical neighbors (100% spatial info)
2. 24x24 = 576 points reduces compression ratio from 1:327 to 1:57
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
from src.models.grid_segnet import GridSegNet, Conv2dEncoderConfig, Decoder3dConfig
from src.models.losses import dice_loss
from src.utils.config import load_yaml


def preprocess_to_grid(signals, traj, grid_size=24):
    """
    Convert signal + trajectory to 2D grid input.
    signals: (B, sweep_steps, samples)
    traj: (B, sweep_steps, 3)
    Returns: (B, 4, grid_size, grid_size)
    """
    B = signals.size(0)
    device = signals.device
    sweep_steps = signals.size(1)
    side = int(sweep_steps ** 0.5)
    
    # Aggregate signal across samples dimension -> (B, sweep_steps)
    signal_agg = signals.mean(dim=2)  # Average over sample dimension
    
    # Log + MinMax normalization
    signal_agg = torch.sign(signal_agg) * torch.log1p(torch.abs(signal_agg) * 100)
    sig_min = signal_agg.min(dim=1, keepdim=True)[0]
    sig_max = signal_agg.max(dim=1, keepdim=True)[0]
    signal_norm = (signal_agg - sig_min) / (sig_max - sig_min + 1e-8)
    
    # Reshape to 2D grid: (B, side, side)
    signal_2d = signal_norm.view(B, side, side)
    
    # Resize to target grid_size if needed
    if side != grid_size:
        signal_2d = F.interpolate(signal_2d.unsqueeze(1), size=(grid_size, grid_size), mode='bilinear', align_corners=False).squeeze(1)
    
    # Trajectory -> 2D grids for X, Y, Z
    traj_2d = traj.view(B, side, side, 3)  # (B, side, side, 3)
    if side != grid_size:
        traj_2d = traj_2d.permute(0, 3, 1, 2)  # (B, 3, side, side)
        traj_2d = F.interpolate(traj_2d, size=(grid_size, grid_size), mode='bilinear', align_corners=False)
        traj_2d = traj_2d.permute(0, 2, 3, 1)  # (B, grid_size, grid_size, 3)
    
    # Normalize trajectory to [-1, 1]
    traj_norm = traj_2d / 0.2
    
    # Stack: (B, 4, grid_size, grid_size)
    inputs = torch.cat([
        signal_2d.unsqueeze(1),  # (B, 1, H, W)
        traj_norm.permute(0, 3, 1, 2),  # (B, 3, H, W)
    ], dim=1)
    
    return inputs


def main():
    print("=" * 60)
    print("CONV2D + 24x24 GRID OVERFITTING TEST")
    print("(Conv2d sees 100% spatial correlations)")
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
    
    # Check grid size from preprocessed input (C, H, W)
    sample = train_ds[0]
    input_shape = sample["signals"].shape  # (4, H, W)
    grid_side = input_shape[1]  # H dimension
    print(f"   Input shape: {input_shape} (grid {grid_side}x{grid_side})")
    
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    
    # Build model
    print("\n2. Building GridSegNet (Conv2d)...")
    encoder_cfg = Conv2dEncoderConfig(base_channels=64, depth=3, latent_dim=512)
    decoder_cfg = Decoder3dConfig(base_channels=64, depth=4, output_size=(64, 64, 64))
    model = GridSegNet(encoder_cfg, decoder_cfg, grid_size=grid_side)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Loss and optimizer
    pos_weight = torch.tensor([20.0]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    
    print("\n3. Training for 100 epochs...")
    print("-" * 60)
    
    best_dice = 0.0
    
    for epoch in range(100):
        model.train()
        epoch_dice = 0.0
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in loader:
            # Dataset already preprocesses to (B, 4, H, W) format
            inputs = batch["signals"].to(device)
            mask = batch["mask"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
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
                d = (2 * intersection / (union + 1e-6)).item()
            
            epoch_loss += loss.item()
            epoch_dice += d
            n_batches += 1
        
        scheduler.step()
        avg_dice = epoch_dice / n_batches
        avg_loss = epoch_loss / n_batches
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), "outputs/grid_segnet_best.pth")
        
        if epoch % 10 == 0 or epoch == 99:
            print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}, dice={avg_dice:.4f}, best={best_dice:.4f}")
    
    print("\n" + "=" * 60)
    print(f"Final Best Dice: {best_dice:.4f}")
    if best_dice > 0.8:
        print("✅ SUCCESS: Conv2d + Dense Grid achieved high Dice!")
    elif best_dice > 0.7:
        print("⚠️  GOOD: Significant improvement, may need more epochs.")
    else:
        print("❌ FAIL: Still room for improvement.")


if __name__ == "__main__":
    main()
