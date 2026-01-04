#!/usr/bin/env python
"""
Overfitting Test with Simple Encoder-Decoder Architecture.
Goal: Verify the model can memorize 500 samples.

Steps:
1. Generate small dataset with large objects
2. Train SimpleSegNet for 50 epochs
3. Expect loss to approach 0.1

Usage:
    python tests/overfit_simple.py --generate  # Generate data first
    python tests/overfit_simple.py             # Train
"""
import argparse
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


def generate_overfit_data():
    """Generate small dataset for overfitting."""
    print("Generating overfit dataset...")
    import subprocess
    cmd = [
        sys.executable, "scripts/gen_data.py",
        "--config", "configs/data_overfit.yaml",
        "--device", "cuda"
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print("Done!")


def train():
    print("=" * 60)
    print("SIMPLE SEGNET OVERFITTING TEST")
    print("500 samples, large objects, 50 epochs")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load dataset
    print("\n1. Loading dataset...")
    data_cfg = load_yaml("configs/data_overfit.yaml")
    
    train_dir = Path(data_cfg["dataset"]["output_dir"]) / "train"
    if not train_dir.exists():
        print(f"   Dataset not found at {train_dir}")
        print("   Run with --generate flag first!")
        return
    
    shard_size = data_cfg["dataset"].get("shard_size")
    train_ds = VolumeDataset(str(train_dir), shard_size=shard_size)
    print(f"   Samples: {len(train_ds)}")
    
    # Check data properties
    sample = train_ds[0]
    signals = sample["signals"]
    mask = sample["mask"]
    
    print(f"   Signals: shape={signals.shape}, range=[{signals.min():.2f}, {signals.max():.2f}]")
    print(f"   Mask: shape={mask.shape}, positive ratio={(mask > 0).float().mean().item() * 100:.1f}%")
    
    # DataLoader
    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    
    # Build model
    print("\n2. Building SimpleSegNet...")
    encoder_cfg = SimpleEncoderConfig(hidden=256, layers=4, latent_dim=512)
    decoder_cfg = SimpleDecoderConfig(
        base_channels=64, 
        depth=4, 
        grid_shape=tuple(data_cfg["grid"]["size"])
    )
    model = SimpleSegNet(encoder_cfg, decoder_cfg)
    model.build(signals.shape)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    
    # Loss and optimizer
    pos_weight = torch.tensor([50.0]).to(device)  # Higher weight for positives
    
    def combo_loss(logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        dice = dice_loss(logits, target)
        return 0.5 * bce + 0.5 * dice
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Training
    print("\n3. Training for 50 epochs...")
    print("-" * 60)
    
    best_loss = float("inf")
    best_dice = 0.0
    
    for epoch in range(50):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch:2d}", leave=False)
        for batch in pbar:
            signals = batch["signals"].to(device)
            mask = batch["mask"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(signals)
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
            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "dice": best_dice,
            }, "outputs/overfit_best.pth")
        
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}, dice={avg_dice:.4f}, lr={lr:.2e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("OVERFITTING RESULTS")
    print("=" * 60)
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Best Dice: {best_dice:.4f}")
    print()
    
    if best_loss < 0.2 and best_dice > 0.7:
        print("✅ SUCCESS: Model can overfit!")
        print("   - Architecture is working")
        print("   - Data pipeline is correct")
        print("   - Ready to scale up")
    elif best_loss < 0.4:
        print("⚠️  PARTIAL: Model is learning")
        print("   - Try: more epochs, higher LR")
    else:
        print("❌ FAIL: Model cannot overfit")
        print("   - Check architecture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate overfit dataset")
    args = parser.parse_args()
    
    if args.generate:
        generate_overfit_data()
    else:
        train()
