#!/usr/bin/env python
"""
Extended Overfitting with 32x32 Grid.
- Trains for 500 epochs
- If loss > 0.1 after 300 epochs, duplicates dataset 4x and continues
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from src.data.dataset import VolumeDataset
from src.models.grid_segnet import GridSegNet, Conv2dEncoderConfig, Decoder3dConfig
from src.models.losses import dice_loss
from src.utils.config import load_yaml


def preprocess_to_grid(signals, traj, grid_size=32):
    B = signals.size(0)
    device = signals.device
    sweep_steps = signals.size(1)
    side = int(sweep_steps ** 0.5)
    
    signal_agg = signals.mean(dim=2)
    signal_agg = torch.sign(signal_agg) * torch.log1p(torch.abs(signal_agg) * 100)
    sig_min = signal_agg.min(dim=1, keepdim=True)[0]
    sig_max = signal_agg.max(dim=1, keepdim=True)[0]
    signal_norm = (signal_agg - sig_min) / (sig_max - sig_min + 1e-8)
    
    signal_2d = signal_norm.view(B, side, side)
    if side != grid_size:
        signal_2d = F.interpolate(signal_2d.unsqueeze(1), size=(grid_size, grid_size), mode='bilinear', align_corners=False).squeeze(1)
    
    traj_2d = traj.view(B, side, side, 3)
    if side != grid_size:
        traj_2d = traj_2d.permute(0, 3, 1, 2)
        traj_2d = F.interpolate(traj_2d, size=(grid_size, grid_size), mode='bilinear', align_corners=False)
        traj_2d = traj_2d.permute(0, 2, 3, 1)
    
    traj_norm = traj_2d / 0.2
    
    inputs = torch.cat([
        signal_2d.unsqueeze(1),
        traj_norm.permute(0, 3, 1, 2),
    ], dim=1)
    
    return inputs


def train_epoch(model, loader, optimizer, device, pos_weight, grid_size):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    n_batches = 0
    
    for batch in loader:
        signals = batch["signals"].to(device)
        traj = batch["traj"].to(device)
        mask = batch["mask"].to(device)
        
        inputs = preprocess_to_grid(signals, traj, grid_size=grid_size)
        
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
    
    return epoch_loss / n_batches, epoch_dice / n_batches


def main():
    print("=" * 60)
    print("EXTENDED 32x32 GRID OVERFITTING")
    print("500 epochs, sample duplication if loss > 0.1")
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
    
    sweep_steps = train_ds[0]["signals"].shape[0]
    grid_side = int(sweep_steps ** 0.5)
    print(f"   Grid: {grid_side}x{grid_side} = {sweep_steps} points")
    
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    
    # Build model
    print("\n2. Building GridSegNet (Conv2d, 32x32)...")
    encoder_cfg = Conv2dEncoderConfig(base_channels=64, depth=4, latent_dim=512)
    decoder_cfg = Decoder3dConfig(base_channels=64, depth=4, output_size=(64, 64, 64))
    model = GridSegNet(encoder_cfg, decoder_cfg, grid_size=grid_side)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    pos_weight = torch.tensor([20.0]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)
    
    print("\n3. Training for 500 epochs...")
    print("-" * 60)
    
    best_loss = float("inf")
    best_dice = 0.0
    duplicated = False
    
    for epoch in range(500):
        avg_loss, avg_dice = train_epoch(model, loader, optimizer, device, pos_weight, grid_side)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dice = avg_dice
            torch.save(model.state_dict(), "outputs/grid32_best.pth")
        
        if epoch % 20 == 0 or epoch == 499:
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, dice={avg_dice:.4f}, best_loss={best_loss:.4f}, best_dice={best_dice:.4f}")
        
        # Check at epoch 300: if loss > 0.1, duplicate dataset
        if epoch == 300 and best_loss > 0.1 and not duplicated:
            print("\n⚠️  Loss > 0.1 at epoch 300, duplicating dataset 4x...")
            dup_ds = ConcatDataset([train_ds, train_ds, train_ds, train_ds])
            loader = DataLoader(dup_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
            print(f"   New dataset size: {len(dup_ds)}")
            duplicated = True
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Best Dice: {best_dice:.4f}")
    
    if best_loss < 0.1:
        print("✅ SUCCESS: Loss < 0.1 achieved!")
    elif best_dice > 0.85:
        print("✅ SUCCESS: Dice > 0.85 achieved!")
    else:
        print("⚠️  Need more optimization.")


if __name__ == "__main__":
    main()
