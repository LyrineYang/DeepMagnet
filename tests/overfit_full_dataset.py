#!/usr/bin/env python
"""
Overfitting Test on Full Dataset (Train + Val + Test combined).
This tests model capacity - if loss doesn't approach 0, model is too small.

Expected theoretical loss:
- Dice Loss: Can reach ~0.05 (95% overlap)
- Weighted BCE: Can reach ~0.01
- Combined: Can reach ~0.1
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
from src.data.shapes import create_grid
from src.models.deeponet import BranchConfig, DeepONet, TrunkConfig
from src.models.losses import LossComputer, dice_loss
from src.models.metrics import compute_metrics
from src.utils.config import load_yaml


def main():
    print("=" * 60)
    print("FULL DATASET OVERFITTING TEST")
    print("Train + Val + Test combined")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load configs
    data_cfg = load_yaml("configs/data_h100.yaml")
    model_cfg = load_yaml("configs/model.yaml")
    
    # Combine all datasets
    print("\n1. Loading all datasets...")
    shard_size = data_cfg["dataset"].get("shard_size")
    train_ds = VolumeDataset("data/processed/train", shard_size=shard_size)
    val_ds = VolumeDataset("data/processed/val", shard_size=shard_size)
    test_ds = VolumeDataset("data/processed/test", shard_size=shard_size)
    
    full_ds = ConcatDataset([train_ds, val_ds, test_ds])
    print(f"   Total samples: {len(full_ds)} (Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)})")
    
    # DataLoader with small batch for memory
    loader = DataLoader(full_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    
    # Create grid
    grid_shape = tuple(data_cfg["grid"]["size"])
    grid_bounds = tuple(data_cfg["grid"]["bounds"])
    grid_coords = create_grid(grid_shape, grid_bounds, device)
    flat_coords = grid_coords.view(-1, 3)
    
    # Build model
    print("\n2. Building model...")
    sample = full_ds[0]
    signal_shape = sample["signals"].shape
    
    branch_cfg = BranchConfig(
        hidden=model_cfg["arch"]["branch"]["hidden"],
        layers=model_cfg["arch"]["branch"]["layers"],
        kernel_size=model_cfg["arch"]["branch"]["kernel_size"],
    )
    trunk_cfg = TrunkConfig(
        hidden=model_cfg["arch"]["trunk"]["hidden"],
        layers=model_cfg["arch"]["trunk"]["layers"],
    )
    model = DeepONet(branch_cfg, trunk_cfg, out_channels=1)
    model.build(signal_shape)
    model = model.to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    loss_fn = LossComputer(mask_weight=1.0, b_weight=0.0, pos_weight=100.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    
    # Training loop - 20 epochs
    print("\n3. Overfitting for 20 epochs...")
    print("-" * 60)
    
    best_loss = float("inf")
    best_dice = 0.0
    
    for epoch in range(20):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            signals = batch["signals"].to(device)
            mask = batch["mask"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(signals, flat_coords)
            outputs["mask_logits"] = outputs["mask_logits"].view(signals.size(0), *grid_shape)
            
            losses = loss_fn(outputs, {"mask": mask})
            loss = losses["total"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            with torch.no_grad():
                metrics = compute_metrics(outputs, {"mask": mask}, thresh=0.5)
                dice = metrics.get("dice", torch.tensor(0.0)).item()
            
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
        
        print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}, dice={avg_dice:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("OVERFITTING RESULTS")
    print("=" * 60)
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Best Dice: {best_dice:.4f}")
    print()
    
    if best_loss < 0.3:
        print("✅ Model has sufficient capacity!")
        print("   - Loss can approach 0 with more training")
        print("   - Focus on regularization for generalization")
    elif best_loss < 0.5:
        print("⚠️  Model is learning but slowly")
        print("   - Consider: more epochs, higher LR, larger model")
    else:
        print("❌ Model struggles to fit training data")
        print("   - Consider: simpler data, larger model, better features")
    
    print("-" * 60)
    print(f"Theoretical minimum with perfect predictions:")
    print(f"  - Dice Loss: ~0 (perfect overlap)")
    print(f"  - BCE Loss: ~0 (perfect classification)")
    print(f"  - Combined: ~0.05-0.1 (practical minimum)")


if __name__ == "__main__":
    main()
