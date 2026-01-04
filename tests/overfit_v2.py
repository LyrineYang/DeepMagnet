#!/usr/bin/env python
"""
Simplified Overfitting Test with pure Weighted BCE.
This isolates the loss function from architecture issues.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.dataset import VolumeDataset
from src.data.shapes import create_grid
from src.models.deeponet import BranchConfig, DeepONet, TrunkConfig
from src.utils.config import load_yaml


def main():
    print("=" * 60)
    print("OVERFITTING TEST v2: Pure Weighted BCE")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load configs
    data_cfg = load_yaml("configs/data_h100.yaml")
    model_cfg = load_yaml("configs/model.yaml")
    
    # Load a tiny subset: just 4 samples (even smaller)
    print("\n1. Loading 4 samples from dataset...")
    shard_size = data_cfg["dataset"].get("shard_size")
    full_ds = VolumeDataset("data/processed/train", shard_size=shard_size)
    tiny_ds = Subset(full_ds, list(range(4)))
    
    loader = DataLoader(tiny_ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    
    print(f"   Signals shape: {batch['signals'].shape}")
    print(f"   Signals range: [{batch['signals'].min():.2f}, {batch['signals'].max():.2f}]")
    print(f"   Mask sparsity: {(batch['mask'] == 0).float().mean().item() * 100:.1f}% zeros")
    
    # Move to device
    signals = batch["signals"].to(device)
    mask = batch["mask"].to(device)
    
    # Create grid
    grid_shape = tuple(data_cfg["grid"]["size"])
    grid_bounds = tuple(data_cfg["grid"]["bounds"])
    grid_coords = create_grid(grid_shape, grid_bounds, device)
    flat_coords = grid_coords.view(-1, 3)
    
    # Build model
    print("\n2. Building model...")
    signal_shape = signals.shape[1:]
    branch_cfg = BranchConfig(
        hidden=model_cfg["arch"]["branch"]["hidden"],
        layers=model_cfg["arch"]["branch"]["layers"],
        kernel_size=model_cfg["arch"]["branch"]["kernel_size"],
    )
    trunk_cfg = TrunkConfig(
        hidden=model_cfg["arch"]["trunk"]["hidden"],
        layers=model_cfg["arch"]["trunk"]["layers"],
    )
    model = DeepONet(branch_cfg, trunk_cfg, out_channels=1)  # Only mask head
    model.build(signal_shape)
    model = model.to(device)
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use Weighted BCE: pos_weight=100 since 99.5% background
    pos_weight = torch.tensor([100.0]).to(device)  # Higher weight for positives
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer with higher LR and no weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # Even higher LR
    
    # Training loop
    print("\n3. Overfitting on single batch (500 iterations)...")
    print("-" * 60)
    model.train()
    
    for step in range(500):
        optimizer.zero_grad()
        
        # Forward
        outputs = model(signals, flat_coords)
        mask_logits = outputs["mask_logits"].view(signals.size(0), *grid_shape)
        
        # Pure Weighted BCE Loss
        loss = bce_loss_fn(mask_logits, mask)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        if step % 50 == 0 or step == 499:
            with torch.no_grad():
                probs = torch.sigmoid(mask_logits)
                pred = (probs > 0.5).float()
                
                # Simple IoU
                intersection = (pred * mask).sum()
                union = pred.sum() + mask.sum() - intersection
                iou = (intersection / (union + 1e-6)).item()
                
                # Dice
                dice = (2 * intersection / (pred.sum() + mask.sum() + 1e-6)).item()
                
                pred_mean = probs.mean().item()
                pred_max = probs.max().item()
                pred_count = (probs > 0.5).sum().item()
                gt_count = mask.sum().item()
                
            print(f"Step {step:3d}: loss={loss.item():.4f}, dice={dice:.4f}, iou={iou:.4f}, "
                  f"pred_count={pred_count:.0f}/{gt_count:.0f}, pred_max={pred_max:.4f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        outputs = model(signals, flat_coords)
        mask_logits = outputs["mask_logits"].view(signals.size(0), *grid_shape)
        probs = torch.sigmoid(mask_logits)
        pred = (probs > 0.5).float()
        
        for i in range(signals.size(0)):
            gt_positive = mask[i].sum().item()
            pred_positive = pred[i].sum().item()
            intersection = (pred[i] * mask[i]).sum().item()
            recall = intersection / (gt_positive + 1e-6)
            precision = intersection / (pred_positive + 1e-6)
            
            print(f"Sample {i}: GT={gt_positive:.0f}, Pred={pred_positive:.0f}, "
                  f"Intersection={intersection:.0f}, Recall={recall:.2f}, Precision={precision:.2f}")
    
    final_dice = dice
    
    print("\n" + "-" * 60)
    if final_dice > 0.3:
        print("✅ PASS: Model can overfit with pure BCE!")
        print("   The architecture works. Focus on loss function tuning.")
    elif final_dice > 0.05:
        print("⚠️  PARTIAL: Model is learning slowly.")
        print("   Try: higher learning rate, longer training, or simpler model.")
    else:
        print("❌ FAIL: Model cannot learn even with simplified loss.")
        print("   Check: architecture gradient flow, data normalization.")
    print("-" * 60)


if __name__ == "__main__":
    main()
