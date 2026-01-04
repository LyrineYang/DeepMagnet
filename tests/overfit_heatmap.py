#!/usr/bin/env python
"""
Overfitting Test with 2D Heatmap Input + Position Encoding.

Implements three key improvements:
1. Reshape 1D signal (64,256) -> 2D heatmap (32,32) for spatial alignment
2. Add trajectory XY as position encoding channels
3. Strong signal normalization (log + minmax)

Expected: Dice 0.28 -> 0.85+
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


def preprocess_sample(signals, traj, grid_size=32):
    """
    Convert 1D signal + trajectory to 2D heatmap with position encoding.
    
    Args:
        signals: (B, 64, 256) raw signal
        traj: (B, 64, 3) trajectory coordinates
        grid_size: output heatmap size (32x32)
    
    Returns:
        heatmap: (B, 3, grid_size, grid_size)
            - Channel 0: Signal intensity (normalized)
            - Channel 1: X position encoding
            - Channel 2: Y position encoding
    """
    B = signals.size(0)
    device = signals.device
    
    # === Plan 3: Strong signal normalization ===
    # Flatten signal to 1D per sample
    signal_flat = signals.view(B, -1)  # (B, 64*256=16384)
    
    # Log transform to amplify weak signals
    signal_flat = torch.sign(signal_flat) * torch.log1p(torch.abs(signal_flat) * 100)
    
    # Per-sample min-max normalization to [0, 1]
    sig_min = signal_flat.min(dim=1, keepdim=True)[0]
    sig_max = signal_flat.max(dim=1, keepdim=True)[0]
    signal_norm = (signal_flat - sig_min) / (sig_max - sig_min + 1e-8)
    
    # === Plan 1: Reshape to 2D heatmap ===
    # Downsample from 16384 to 32*32=1024 via average pooling
    target_len = grid_size * grid_size
    pool_size = signal_flat.size(1) // target_len
    signal_2d = signal_norm[:, :pool_size * target_len].view(B, target_len, pool_size).mean(dim=2)
    signal_2d = signal_2d.view(B, 1, grid_size, grid_size)  # (B, 1, 32, 32)
    
    # === Plan 2: Position encoding from trajectory ===
    # Take trajectory XY and expand to 2D grid
    # Trajectory is (B, 64, 3), we need (B, 32, 32) position maps
    traj_x = traj[:, :, 0]  # (B, 64)
    traj_y = traj[:, :, 1]  # (B, 64)
    
    # Normalize trajectory to [0, 1]
    traj_x = (traj_x - traj_x.min(dim=1, keepdim=True)[0]) / (traj_x.max(dim=1, keepdim=True)[0] - traj_x.min(dim=1, keepdim=True)[0] + 1e-8)
    traj_y = (traj_y - traj_y.min(dim=1, keepdim=True)[0]) / (traj_y.max(dim=1, keepdim=True)[0] - traj_y.min(dim=1, keepdim=True)[0] + 1e-8)
    
    # Create meshgrid-style position encoding
    # For raster scan assumption: positions form a regular grid
    x_pos = torch.linspace(0, 1, grid_size, device=device).view(1, 1, 1, grid_size).expand(B, 1, grid_size, grid_size)
    y_pos = torch.linspace(0, 1, grid_size, device=device).view(1, 1, grid_size, 1).expand(B, 1, grid_size, grid_size)
    
    # Concatenate: signal + position encoding
    heatmap = torch.cat([signal_2d, x_pos, y_pos], dim=1)  # (B, 3, 32, 32)
    
    return heatmap


def main():
    print("=" * 60)
    print("HEATMAP OVERFITTING TEST")
    print("2D Signal Heatmap + Position Encoding + Strong Normalization")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load overfit dataset (500 samples, larger objects)
    print("\n1. Loading dataset...")
    data_cfg = load_yaml("configs/data_overfit.yaml")
    train_dir = Path(data_cfg["dataset"]["output_dir"]) / "train"
    
    if not train_dir.exists():
        print(f"   Dataset not found! Run: python tests/overfit_simple.py --generate")
        return
    
    shard_size = data_cfg["dataset"].get("shard_size")
    train_ds = VolumeDataset(str(train_dir), shard_size=shard_size)
    print(f"   Samples: {len(train_ds)}")
    
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    
    # Build model
    print("\n2. Building HeatmapSegNet...")
    encoder_cfg = HeatmapEncoderConfig(base_channels=64, depth=4)
    decoder_cfg = VolumeDecoderConfig(base_channels=64, depth=5, output_size=(64, 64, 64))
    model = HeatmapSegNet(encoder_cfg, decoder_cfg)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Test preprocessing
    sample = train_ds[0]
    heatmap = preprocess_sample(
        sample["signals"].unsqueeze(0), 
        sample["traj"].unsqueeze(0)
    )
    print(f"   Input heatmap shape: {heatmap.shape}")
    
    # Loss and optimizer
    pos_weight = torch.tensor([30.0]).to(device)
    
    def combo_loss(logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        dice = dice_loss(logits, target)
        return 0.5 * bce + 0.5 * dice
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # Training for 100 epochs
    print("\n3. Training for 100 epochs...")
    print("-" * 60)
    
    best_loss = float("inf")
    best_dice = 0.0
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch:2d}", leave=False)
        for batch in pbar:
            signals = batch["signals"].to(device)
            mask = batch["mask"].to(device)
            traj = batch["traj"].to(device)
            
            # Preprocess to heatmap
            heatmap = preprocess_sample(signals, traj)
            
            optimizer.zero_grad()
            
            outputs = model(heatmap)
            logits = outputs["mask_logits"]
            
            loss = combo_loss(logits, mask)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
                intersection = (pred * mask).sum()
                union = pred.sum() + mask.sum()
                dice = (2 * intersection / (union + 1e-6)).item()
            
            epoch_loss += loss.item()
            epoch_dice += dice
            n_batches += 1
            
            pbar.set_postfix({"loss": loss.item(), "dice": dice})
        
        scheduler.step()
        
        avg_loss = epoch_loss / n_batches
        avg_dice = epoch_dice / n_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dice = avg_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "dice": best_dice,
            }, "outputs/heatmap_best.pth")
        
        lr = scheduler.get_last_lr()[0]
        
        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 99:
            print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}, dice={avg_dice:.4f}, best_dice={best_dice:.4f}, lr={lr:.2e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("OVERFITTING RESULTS")
    print("=" * 60)
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Best Dice: {best_dice:.4f}")
    print()
    
    if best_dice > 0.8:
        print("✅ SUCCESS: Model can overfit with 2D heatmap approach!")
        print("   Ready to scale up to full dataset.")
    elif best_dice > 0.5:
        print("⚠️  PARTIAL: Significant improvement over baseline.")
        print("   Try: more epochs, larger model, better preprocessing.")
    else:
        print("❌ FAIL: Still struggling to overfit.")
        print("   May need more fundamental changes.")


if __name__ == "__main__":
    main()
