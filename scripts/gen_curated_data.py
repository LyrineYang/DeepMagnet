#!/usr/bin/env python
"""
Generate a curated, diverse mini-dataset for overfitting.
Ensures: 20 MNIST (2 per digit 0-9) + 15 boxes + 15 cylinders = 50 samples
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from src.data.coil import CoilConfig, sample_trajectory
from src.data.physics import GridConfig, field_volume, synthesize_signal
from src.data.shapes import MnistConfig, ShapeConfig, MnistProvider, create_grid, box_mask, cylinder_mask, random_center
from src.utils.config import load_yaml
from src.utils.io import save_shard

try:
    from torchvision import datasets, transforms
except ImportError:
    raise ImportError("torchvision is required")


def get_mnist_by_label(mnist_ds, label: int, device: torch.device) -> torch.Tensor:
    """Get a random sample of a specific digit (0-9)."""
    indices = [i for i, (_, lbl) in enumerate(mnist_ds) if lbl == label]
    idx = indices[torch.randint(0, len(indices), (1,)).item()]
    img, _ = mnist_ds[idx]
    return img.squeeze(0).to(device)


def generate_mnist_sample(digit: int, grid_coords: torch.Tensor, mnist_ds, 
                          coil_cfg: CoilConfig, mn_cfg: MnistConfig, device: torch.device):
    """Generate a sample for a specific MNIST digit."""
    import torch.nn.functional as F
    
    img = get_mnist_by_label(mnist_ds, digit, device)
    img = (img > mn_cfg.threshold).float()
    
    # Apply dilation
    if mn_cfg.dilation > 0:
        kernel_size = 2 * mn_cfg.dilation + 1
        img = F.max_pool2d(img.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=mn_cfg.dilation).squeeze()
    
    # Scale to target size
    target_hw = int(grid_coords.shape[0] * 0.7)  # Fixed 70% scale for consistency
    img = F.interpolate(img.view(1, 1, *img.shape), size=(target_hw, target_hw), mode="bilinear", align_corners=False).squeeze()
    
    # Center the digit in the grid
    margin = 2
    x0 = (grid_coords.shape[0] - target_hw) // 2
    y0 = (grid_coords.shape[1] - target_hw) // 2
    depth = int(grid_coords.shape[2] * mn_cfg.extrude_depth)
    z0 = (grid_coords.shape[2] - depth) // 2  # Center in Z
    
    mask = torch.zeros(grid_coords.shape[:-1], device=device)
    digit_3d = img.clamp(0, 1).unsqueeze(-1).expand(target_hw, target_hw, depth)
    x1 = min(x0 + target_hw, mask.shape[0])
    y1 = min(y0 + target_hw, mask.shape[1])
    z1 = min(z0 + depth, mask.shape[2])
    mask[x0:x1, y0:y1, z0:z1] = digit_3d[: x1 - x0, : y1 - y0, : z1 - z0]
    
    traj = sample_trajectory(coil_cfg, device)
    signals = synthesize_signal(coil_cfg, traj, mask, grid_coords)
    bfield = field_volume(coil_cfg, traj_idx=0, traj=traj, grid_coords=grid_coords)
    
    return {
        "signals": signals.float(),
        "mask": mask.float(),
        "bfield": bfield.float(),
        "traj": traj.float(),
        "meta": {"type": "mnist", "digit": digit},
    }


def generate_shape_sample(shape_type: str, grid_coords: torch.Tensor, 
                          coil_cfg: CoilConfig, size_range: tuple, device: torch.device):
    """Generate a sample for box or cylinder."""
    import random
    bounds = (grid_coords[..., 0].min().item(), grid_coords[..., 0].max().item())
    size = random.uniform(size_range[0], size_range[1])
    center = random_center(bounds, 0.01, device)
    
    if shape_type == "box":
        mask = box_mask(grid_coords, center, size)
    else:  # cylinder
        mask = cylinder_mask(grid_coords, center, size / 2, size)
    
    traj = sample_trajectory(coil_cfg, device)
    signals = synthesize_signal(coil_cfg, traj, mask, grid_coords)
    bfield = field_volume(coil_cfg, traj_idx=0, traj=traj, grid_coords=grid_coords)
    
    return {
        "signals": signals.float(),
        "mask": mask.float(),
        "bfield": bfield.float(),
        "traj": traj.float(),
        "meta": {"type": shape_type, "center": center.tolist(), "size": size},
    }


def main(args):
    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    
    # Setup configs
    grid_cfg = GridConfig(
        size=tuple(cfg["grid"]["size"]),
        bounds=tuple(cfg["grid"]["bounds"]),
        voxel_size=cfg["grid"]["voxel_size"]
    )
    coil_cfg = CoilConfig(
        type=cfg["coil"]["type"],
        radius=cfg["coil"]["radius"],
        separation=cfg["coil"]["separation"],
        turns=cfg["coil"]["turns"],
        current=cfg["coil"]["current"],
        frequency=cfg["coil"]["frequency"],
        sweep_steps=cfg["coil"]["sweep_steps"],
        trajectory=cfg["coil"]["trajectory"],
        samples=cfg["signal"]["samples"],
        noise_std=cfg["signal"]["noise_std"],
    )
    mn_cfg = MnistConfig(
        root=cfg["shapes"]["mnist"].get("root", "data/mnist"),
        threshold=cfg["shapes"]["mnist"].get("threshold", 0.3),
        scale_range=tuple(cfg["shapes"]["mnist"].get("scale_range", [0.5, 0.8])),
        extrude_depth=cfg["shapes"]["mnist"].get("extrude_depth", 0.25),
        rotate=False,  # Fixed for overfitting
        dilation=cfg["shapes"]["mnist"].get("dilation", 2),
    )
    
    grid_coords = create_grid(grid_cfg.size, grid_cfg.bounds, device)
    
    # Load MNIST
    mnist_ds = datasets.MNIST(mn_cfg.root, train=True, download=True, transform=transforms.ToTensor())
    
    # Output directory
    out_dir = Path(cfg["dataset"]["output_dir"])
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    # === CURATED GENERATION ===
    print("Generating curated dataset...")
    
    # 1. MNIST: 2 samples per digit (0-9) = 20 samples
    print("  MNIST digits (20 samples):")
    for digit in tqdm(range(10), desc="  Digits"):
        for _ in range(2):  # 2 samples per digit
            sample = generate_mnist_sample(digit, grid_coords, mnist_ds, coil_cfg, mn_cfg, device)
            samples.append(sample)
    
    # 2. Boxes: 15 samples
    print("  Boxes (15 samples):")
    for _ in tqdm(range(15), desc="  Boxes"):
        sample = generate_shape_sample("box", grid_coords, coil_cfg, 
                                        tuple(cfg["shapes"]["size_range"]), device)
        samples.append(sample)
    
    # 3. Cylinders: 15 samples
    print("  Cylinders (15 samples):")
    for _ in tqdm(range(15), desc="  Cylinders"):
        sample = generate_shape_sample("cylinder", grid_coords, coil_cfg, 
                                        tuple(cfg["shapes"]["size_range"]), device)
        samples.append(sample)
    
    # Shuffle and split
    import random
    random.shuffle(samples)
    
    train_samples = samples[:45]  # 45 for training
    val_samples = samples[45:]    # 5 for validation
    
    # Save train shard
    train_batch = {
        "signals": torch.stack([s["signals"] for s in train_samples]),
        "mask": torch.stack([s["mask"] for s in train_samples]),
        "bfield": torch.stack([s["bfield"] for s in train_samples]),
        "traj": torch.stack([s["traj"] for s in train_samples]),
    }
    train_metas = [s["meta"] for s in train_samples]
    save_shard({"tensors": train_batch, "meta": train_metas}, out_dir / "train" / "train_00000.pt")
    
    # Save val shard
    val_batch = {
        "signals": torch.stack([s["signals"] for s in val_samples]),
        "mask": torch.stack([s["mask"] for s in val_samples]),
        "bfield": torch.stack([s["bfield"] for s in val_samples]),
        "traj": torch.stack([s["traj"] for s in val_samples]),
    }
    val_metas = [s["meta"] for s in val_samples]
    save_shard({"tensors": val_batch, "meta": val_metas}, out_dir / "val" / "val_00000.pt")
    
    print(f"\n✅ Done! Generated {len(train_samples)} train + {len(val_samples)} val samples")
    print(f"   Train: 18 MNIST + 13 boxes + 14 cylinders (approx)")
    print(f"   Val: 2 MNIST + 2 boxes + 1 cylinder (approx)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_overfit.yaml")
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
