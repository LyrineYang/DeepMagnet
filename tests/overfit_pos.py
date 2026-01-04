#!/usr/bin/env python
"""
Overfitting Test with Positional Encoding (GPS).
Inputs: (B, 4, Length) -> Output: (B, 1, D, H, W)
Channels:
0: Signal (Log-Normalized)
1: Trajectory X
2: Trajectory Y
3: Trajectory Z

This tells the model EXACTLY where the sensor is for every voltage reading.
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
from src.models.simple_segnet import SimpleSegNet, SimpleEncoderConfig, SimpleDecoderConfig
from src.models.losses import dice_loss
from src.utils.config import load_yaml


def preprocess_sample(signals, traj):
    """
    Combine signal and trajectory into 4-channel input.
    signals: (B, 64, 256)
    traj: (B, 64, 3)
    Returns: (B, 4, 16384)
    """
    B = signals.size(0)
    device = signals.device
    
    # 1. Flatten signal: (B, 16384)
    # Log-normalize signal
    signal_flat = signals.view(B, -1)
    signal_flat = torch.sign(signal_flat) * torch.log1p(torch.abs(signal_flat) * 100)
    
    # Min-max norm per sample
    sig_min = signal_flat.min(dim=1, keepdim=True)[0]
    sig_max = signal_flat.max(dim=1, keepdim=True)[0]
    signal_norm = (signal_flat - sig_min) / (sig_max - sig_min + 1e-8)
    
    # 2. Expand trajectory: (B, 64, 3) -> (B, 16384, 3)
    # We assume the sensor moves linearly between steps or stays constant?
    # Usually 'sweep_steps' are measurement points. The '256' samples might be time-series at that point.
    # We'll repeat the trajectory position for all 256 samples at that step.
    traj_expanded = traj.unsqueeze(2).expand(-1, -1, 256, -1)  # (B, 64, 256, 3)
    traj_flat = traj_expanded.reshape(B, -1, 3)  # (B, 16384, 3)
    
    # Normalize trajectory to [-1, 1] (roughly, since bounds are [-0.2, 0.2])
    # Divide by 0.2
    traj_norm = traj_flat / 0.2
    
    # 3. Stack: (B, 16384, 4) -> (B, 4, 16384)
    inputs = torch.cat([signal_norm.unsqueeze(2), traj_norm], dim=2)
    inputs = inputs.permute(0, 2, 1)  # (B, 4, 16384)
    
    return inputs


def main():
    print("=" * 60)
    print("POSITIONAL ENCODING OVERFITTING TEST")
    print("4 Channels: Signal + X + Y + Z")
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
    
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    
    # Build model
    print("\n2. Building SimpleSegNet (4-channel)...")
    encoder_cfg = SimpleEncoderConfig(hidden=256, layers=4, latent_dim=512, input_channels=4)
    decoder_cfg = SimpleDecoderConfig(base_channels=64, depth=4, grid_shape=(64, 64, 64))
    model = SimpleSegNet(encoder_cfg, decoder_cfg)
    model = model.to(device)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
        
        # pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch in loader:
            signals = batch["signals"].to(device)
            traj = batch["traj"].to(device)
            mask = batch["mask"].to(device)
            
            inputs = preprocess_sample(signals, traj)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs["mask_logits"]
            
            bce = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight)
            dice = dice_loss(logits, mask)
            loss = 0.5 * bce + 0.5 * dice
            
            loss.backward()
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
            
        if epoch % 5 == 0 or epoch == 49:
            print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}, dice={avg_dice:.4f}, best={best_dice:.4f}")
            
    print("\n" + "=" * 60)
    print(f"Final Best Dice: {best_dice:.4f}")
    if best_dice > 0.6:
        print("✅ SUCCESS: Positional Encoding works!")
    else:
        print("❌ FAIL: Still cannot learn geometry.")


if __name__ == "__main__":
    main()
