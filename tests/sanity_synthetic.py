#!/usr/bin/env python
"""
Sanity Check: Can the model learn an identity mapping?
We directly encode mask information into the signal to verify network capacity.

If this fails, the architecture is broken.
If this succeeds, the real signal doesn't contain enough information.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.simple_segnet import SimpleSegNet, SimpleEncoderConfig, SimpleDecoderConfig
from src.models.losses import dice_loss


def create_synthetic_data(n_samples: int = 100, grid_size: int = 64, signal_len: int = 16384):
    """
    Create synthetic data where signal DIRECTLY encodes the mask.
    The signal is a flattened version of the mask, so perfect reconstruction should be possible.
    """
    print(f"Creating {n_samples} synthetic samples...")
    
    signals_list = []
    masks_list = []
    
    for i in range(n_samples):
        # Random box in the volume
        cx = torch.randint(16, 48, (1,)).item()
        cy = torch.randint(16, 48, (1,)).item()
        cz = torch.randint(16, 48, (1,)).item()
        size = torch.randint(10, 20, (1,)).item()
        
        # Create mask
        mask = torch.zeros(grid_size, grid_size, grid_size)
        x0, x1 = max(0, cx - size), min(grid_size, cx + size)
        y0, y1 = max(0, cy - size), min(grid_size, cy + size)
        z0, z1 = max(0, cz - size), min(grid_size, cz + size)
        mask[x0:x1, y0:y1, z0:z1] = 1.0
        
        # Signal encodes mask: downsample mask to signal_len and add small noise
        # This makes the task: "decode the mask from its compressed representation"
        mask_flat = mask.view(-1)  # 262144 values
        # Simple pooling to reduce to signal_len
        pool_size = mask_flat.numel() // signal_len
        signal_1d = mask_flat[:pool_size * signal_len].view(signal_len, pool_size).mean(dim=1)
        
        # Reshape to (64, 256) like real data
        signal = signal_1d.view(64, 256)
        signal = signal + 0.01 * torch.randn_like(signal)  # Small noise
        
        # Normalize
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        
        signals_list.append(signal)
        masks_list.append(mask)
    
    signals = torch.stack(signals_list)
    masks = torch.stack(masks_list)
    
    print(f"  Signals: {signals.shape}, range=[{signals.min():.2f}, {signals.max():.2f}]")
    print(f"  Masks: {masks.shape}, positive ratio={masks.mean().item()*100:.1f}%")
    
    return signals, masks


def main():
    print("=" * 60)
    print("SYNTHETIC SANITY CHECK")
    print("Can the model learn from signal that encodes the mask?")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create synthetic data
    signals, masks = create_synthetic_data(n_samples=100)
    dataset = TensorDataset(signals, masks)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Build model
    print("\nBuilding SimpleSegNet...")
    encoder_cfg = SimpleEncoderConfig(hidden=256, layers=4, latent_dim=512)
    decoder_cfg = SimpleDecoderConfig(base_channels=64, depth=4, grid_shape=(64, 64, 64))
    model = SimpleSegNet(encoder_cfg, decoder_cfg)
    model.build(signals[0].shape)
    model = model.to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    pos_weight = torch.tensor([10.0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    print("\nTraining for 30 epochs on synthetic data...")
    print("-" * 60)
    
    for epoch in range(30):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0
        
        for batch_signals, batch_masks in loader:
            batch_signals = batch_signals.to(device)
            batch_masks = batch_masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_signals)
            logits = outputs["mask_logits"]
            
            bce = F.binary_cross_entropy_with_logits(logits, batch_masks, pos_weight=pos_weight)
            dice = dice_loss(logits, batch_masks)
            loss = 0.5 * bce + 0.5 * dice
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
                intersection = (pred * batch_masks).sum()
                union = pred.sum() + batch_masks.sum()
                dice_score = (2 * intersection / (union + 1e-6)).item()
            
            epoch_loss += loss.item()
            epoch_dice += dice_score
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_dice = epoch_dice / n_batches
        
        if epoch % 5 == 0 or epoch == 29:
            print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}, dice={avg_dice:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    if avg_dice > 0.8:
        print("✅ SUCCESS: Model CAN learn from informative signals!")
        print("   The real issue is: signals don't contain enough mask info.")
    elif avg_dice > 0.5:
        print("⚠️  PARTIAL: Model is learning but slowly.")
        print("   Try: more epochs, different architecture.")
    else:
        print("❌ FAIL: Model cannot learn even from easy synthetic data.")
        print("   Architecture has fundamental issues.")


if __name__ == "__main__":
    main()
