#!/usr/bin/env python
"""
Overfitting Test: Train on a single batch to verify the entire pipeline works.
If loss doesn't approach ~0 after 100+ iterations, there's a fundamental issue.

Usage: python tests/overfit_one_batch.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import VolumeDataset
from src.data.shapes import create_grid
from src.models.deeponet import BranchConfig, DeepONet, TrunkConfig
from src.models.losses import LossComputer, dice_loss, focal_loss
from src.models.metrics import compute_metrics
from src.utils.config import load_yaml


def main():
    print("=" * 60)
    print("OVERFITTING TEST: Single Batch Memorization")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load configs
    data_cfg = load_yaml("configs/data_h100.yaml")
    model_cfg = load_yaml("configs/model.yaml")
    
    # Load a tiny subset: just 8 samples
    print("\n1. Loading 8 samples from dataset...")
    shard_size = data_cfg["dataset"].get("shard_size")
    full_ds = VolumeDataset("data/processed/train", shard_size=shard_size)
    tiny_ds = Subset(full_ds, list(range(8)))  # Only 8 samples
    
    loader = DataLoader(tiny_ds, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    
    print(f"   Signals shape: {batch['signals'].shape}")
    print(f"   Signals range: [{batch['signals'].min():.2f}, {batch['signals'].max():.2f}]")
    print(f"   Mask sparsity: {(batch['mask'] == 0).float().mean().item() * 100:.1f}% zeros")
    
    # Move to device
    signals = batch["signals"].to(device)
    mask = batch["mask"].to(device)
    bfield = batch.get("bfield")
    if bfield is not None:
        bfield = bfield.to(device)
    
    # Create grid
    grid_shape = tuple(data_cfg["grid"]["size"])
    grid_bounds = tuple(data_cfg["grid"]["bounds"])
    grid_coords = create_grid(grid_shape, grid_bounds, device)
    flat_coords = grid_coords.view(-1, 3)
    
    # Build model
    print("\n2. Building model...")
    signal_shape = signals.shape[1:]  # (64, 256)
    branch_cfg = BranchConfig(
        hidden=model_cfg["arch"]["branch"]["hidden"],
        layers=model_cfg["arch"]["branch"]["layers"],
        kernel_size=model_cfg["arch"]["branch"]["kernel_size"],
    )
    trunk_cfg = TrunkConfig(
        hidden=model_cfg["arch"]["trunk"]["hidden"],
        layers=model_cfg["arch"]["trunk"]["layers"],
    )
    model = DeepONet(branch_cfg, trunk_cfg, out_channels=4)
    model.build(signal_shape)
    model = model.to(device)
    
    # Apply bias initialization trick (方案四)
    print("   Applying bias initialization trick (pi=0.01)...")
    import math
    for name, param in model.named_parameters():
        if 'bias' in name and param.numel() == 1:  # Final layer bias
            pi = 0.01
            bias_value = -math.log((1 - pi) / pi)
            param.data.fill_(bias_value)
            print(f"   Set {name} = {bias_value:.4f}")
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with higher LR for overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    loss_fn = LossComputer(mask_weight=1.0, b_weight=0.0)  # Focus only on mask
    
    # Training loop
    print("\n3. Overfitting on single batch (200 iterations)...")
    print("-" * 60)
    model.train()
    
    for step in range(200):
        optimizer.zero_grad()
        
        # Forward
        outputs = model(signals, flat_coords)
        outputs["mask_logits"] = outputs["mask_logits"].view(signals.size(0), *grid_shape)
        
        # Loss
        targets = {"mask": mask}
        losses = loss_fn(outputs, targets)
        loss = losses["total"]
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        if step % 20 == 0 or step == 199:
            with torch.no_grad():
                metrics = compute_metrics(outputs, targets, thresh=0.5)
                dice = metrics.get("dice", torch.tensor(0.0)).item()
                
                # Check prediction distribution
                probs = torch.sigmoid(outputs["mask_logits"])
                pred_mean = probs.mean().item()
                pred_max = probs.max().item()
                
            print(f"Step {step:3d}: loss={loss.item():.4f}, dice={dice:.4f}, "
                  f"pred_mean={pred_mean:.4f}, pred_max={pred_max:.4f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        outputs = model(signals, flat_coords)
        outputs["mask_logits"] = outputs["mask_logits"].view(signals.size(0), *grid_shape)
        
        probs = torch.sigmoid(outputs["mask_logits"])
        pred = (probs > 0.5).float()
        
        # Per-sample analysis
        for i in range(min(4, signals.size(0))):
            gt_positive = mask[i].sum().item()
            pred_positive = pred[i].sum().item()
            intersection = (pred[i] * mask[i]).sum().item()
            
            print(f"Sample {i}: GT positives={gt_positive:.0f}, "
                  f"Pred positives={pred_positive:.0f}, "
                  f"Intersection={intersection:.0f}")
    
    final_loss = loss.item()
    final_dice = dice
    
    print("\n" + "-" * 60)
    if final_loss < 0.5 and final_dice > 0.5:
        print("✅ PASS: Model can overfit a single batch!")
        print("   The pipeline is working correctly.")
    elif final_loss < 0.8:
        print("⚠️  PARTIAL: Loss is decreasing but Dice is low.")
        print("   Model is learning but may need architectural changes.")
    else:
        print("❌ FAIL: Model cannot memorize 8 samples.")
        print("   There's a fundamental issue with the pipeline.")
        print("   Check: data preprocessing, model architecture, loss function.")
    
    print("-" * 60)
    return final_loss, final_dice


if __name__ == "__main__":
    main()
