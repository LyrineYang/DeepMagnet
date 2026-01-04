import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import VolumeDataset
from src.models.deeponet import DeepONet, BranchConfig, TrunkConfig
from src.models.losses import LossComputer
from src.utils.config import load_yaml

def test_training_loop():
    print("1. Loading configs...")
    # Use the same configs as the failed training
    data_cfg = load_yaml("configs/data_h100.yaml")
    train_cfg = load_yaml("configs/train_4gpu.yaml")
    model_cfg = load_yaml("configs/model.yaml")

    print("2. Initializing Dataset...")
    shard_size = data_cfg["dataset"].get("shard_size")
    # Point to the processed train data
    train_ds = VolumeDataset(
        train_cfg["paths"]["train_shards"], 
        shard_size=shard_size
    )
    print(f"   Dataset size: {len(train_ds)}")

    print("3. Initializing DataLoader...")
    # Use num_workers=4 as optimized
    train_loader = DataLoader(
        train_ds, 
        batch_size=4, # Small batch for testing
        shuffle=True, 
        num_workers=4
    )

    print("4. Testing Data Loading (Iterate 5 batches)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    iterator = iter(train_loader)
    for i in range(5):
        try:
            batch = next(iterator)
            print(f"   Batch {i}: signals={batch['signals'].shape}, mask={batch['mask'].shape}")
            # Verify keys
            if "meta" in batch:
                print("   [WARNING] 'meta' key found in batch (Should be removed to avoid collate error)")
            
            # Move to device
            signals = batch["signals"].to(device)
            mask = batch["mask"].to(device)
            bfield = batch.get("bfield").to(device) if "bfield" in batch else None

        except Exception as e:
            print(f"   [FAIL] Data loading failed at batch {i}: {e}")
            raise e

    print("5. Initializing Model...")
    grid_shape = tuple(data_cfg["grid"]["size"])
    signal_shape = train_ds[0]["signals"].shape
    
    branch_cfg = BranchConfig(
        hidden=model_cfg["arch"]["branch"]["hidden"],
        layers=model_cfg["arch"]["branch"]["layers"],
        kernel_size=model_cfg["arch"]["branch"]["kernel_size"],
    )
    trunk_cfg = TrunkConfig(
        hidden=model_cfg["arch"]["trunk"]["hidden"],
        layers=model_cfg["arch"]["trunk"]["layers"],
    )
    model = DeepONet(branch_cfg, trunk_cfg, out_channels=4).to(device)
    model.build(signal_shape)

    print("6. Testing Forward/Backward Pass...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = LossComputer()
    
    # Create dummy grid coords
    # Note: DeepONet expects flattened coords
    # For test, just random
    flat_coords = torch.randn(grid_shape[0]*grid_shape[1]*grid_shape[2], 3, device=device)

    try:
        # Forward
        outputs = model(signals, flat_coords)
        outputs["mask_logits"] = outputs["mask_logits"].view(signals.size(0), *grid_shape)
        if "bfield" in outputs:
            outputs["bfield"] = outputs["bfield"].view(signals.size(0), *grid_shape, 3).permute(0, 4, 1, 2, 3)
            
        # Target
        targets = {"mask": mask}
        if bfield is not None and "bfield" in outputs:
             targets["bfield"] = bfield.permute(0, 4, 1, 2, 3)

        # Loss
        losses = loss_fn(outputs, targets)
        loss = losses["total"]
        print(f"   Loss value: {loss.item()}")

        # Backward
        loss.backward()
        optimizer.step()
        print("   Backward pass successful.")

    except Exception as e:
        print(f"   [FAIL] Model training step failed: {e}")
        raise e

    print("\n[SUCCESS] Verification Script Passed! The training code path is robust.")

if __name__ == "__main__":
    test_training_loop()
