from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from .coil import CoilConfig, sample_trajectory
from .physics import GridConfig, field_volume, synthesize_signal
from .shapes import MnistProvider, ShapeConfig, create_grid, random_shape
from ..utils.io import save_metadata, save_shard, shard_name, split_counts

@dataclass
class DatasetConfig:
    grid: GridConfig
    coil: CoilConfig
    shapes: ShapeConfig
    dataset: Dict[str, int]


class DataGenerator:
    def __init__(self, grid_cfg: GridConfig, coil_cfg: CoilConfig, shape_cfg: ShapeConfig, device: str = "cpu"):
        self.grid_cfg = grid_cfg
        self.coil_cfg = coil_cfg
        self.shape_cfg = shape_cfg
        self.device = torch.device(device)
        self.grid_coords = create_grid(grid_cfg.size, grid_cfg.bounds, self.device)
        self.mnist_provider = MnistProvider(shape_cfg.mnist) if any(t == "mnist" for t in shape_cfg.types) else None

    def generate_sample(self) -> Dict[str, torch.Tensor | dict]:
        mask, meta = random_shape(self.grid_coords, self.shape_cfg, self.mnist_provider)
        traj = sample_trajectory(self.coil_cfg, self.device)
        signals = synthesize_signal(self.coil_cfg, traj, mask, self.grid_coords)
        bfield = field_volume(self.coil_cfg, traj_idx=0, traj=traj, grid_coords=self.grid_coords)
        return {
            "signals": signals.float(),
            "mask": mask.float(),
            "bfield": bfield.float(),
            "traj": traj.float(),
            "meta": meta,
        }

    def _write_metadata(self, meta_dir: Path) -> None:
        meta = {
            "grid": {
                "size": list(self.grid_cfg.size),
                "bounds": list(self.grid_cfg.bounds),
                "voxel_size": self.grid_cfg.voxel_size,
            },
            "coil": asdict(self.coil_cfg),
            "shapes": asdict(self.shape_cfg),
        }
        save_metadata(meta, meta_dir / "config.yaml")

    def generate_split(self, split: str, total: int, shard_size: int, out_dir: str | Path, meta_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        meta_dir = Path(meta_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        self._write_metadata(meta_dir)

        n_shards, remainder = split_counts(total, shard_size)
        shard_idx = 0
        remaining = total
        progress = tqdm(total=total, desc=f"Generating {split}")
        while remaining > 0:
            current_size = shard_size if remaining >= shard_size else remaining
            batch = {"signals": [], "mask": [], "bfield": [], "traj": [], "meta": []}
            for _ in range(current_size):
                sample = self.generate_sample()
                for k in batch:
                    batch[k].append(sample[k])
                progress.update(1)
            tensors = {
                "signals": torch.stack(batch["signals"], dim=0),
                "mask": torch.stack(batch["mask"], dim=0),
                "bfield": torch.stack(batch["bfield"], dim=0),
                "traj": torch.stack(batch["traj"], dim=0),
            }
            metas = batch["meta"]
            shard_path = out_dir / shard_name(split, shard_idx)
            save_shard({"tensors": tensors, "meta": metas}, shard_path)
            shard_idx += 1
            remaining -= current_size
        progress.close()
