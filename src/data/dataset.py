from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from ..utils.io import load_shard


class VolumeDataset(Dataset):
    def __init__(self, shard_dir: str | Path, load_bfield: bool = True, shard_size: int = None):
        self.shard_dir = Path(shard_dir)
        self.load_bfield = load_bfield
        self.files = sorted(self.shard_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No shards found in {self.shard_dir}")
        self.index_map: List[Tuple[int, int]] = []
        for fi, f in enumerate(self.files):
            # Optimization: If shard_size is provided, assume full shards for all but the last one
            if shard_size is not None and fi < len(self.files) - 1:
                n = shard_size
            else:
                data = load_shard(f)
                n = data["tensors"]["signals"].shape[0]
            self.index_map.extend([(fi, j) for j in range(n)])

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | dict]:
        file_idx, inner_idx = self.index_map[idx]
        shard = load_shard(self.files[file_idx])
        tensors = shard["tensors"]
        
        signals = tensors["signals"][inner_idx]  # (steps, samples)
        traj = tensors["traj"][inner_idx]        # (steps, 3)
        
        # --- Preprocess to 2D Grid (Scheme 1 + 3) ---
        # 1. Aggregate samples (mean) -> (steps,)
        sig_agg = signals.mean(dim=-1)
        
        # 2. Scheme 3: Non-linear Normalization (Log + MinMax)
        # Log scaling to boost weak signals
        sig_log = torch.sign(sig_agg) * torch.log1p(torch.abs(sig_agg) * 100)
        
        # MinMax per sample
        s_min = sig_log.min()
        s_max = sig_log.max()
        sig_norm = (sig_log - s_min) / (s_max - s_min + 1e-8)
        
        # 3. Reshape to 2D Grid
        # Check if steps is a perfect square
        steps = sig_norm.shape[0]
        side = int(steps ** 0.5)
        if side * side != steps:
            # Fallback for non-square (e.g. line trajectory): pad or interpolate?
            # For now, let's assume we fixed the config to 576 (24x24) or similar.
            # If not square, just keep as 1D (train will fail if model expects 2D)
            # But strictly following the plan, we enforce grid structure.
            # Let's try to reshape, or interpolate if 64 steps -> 8x8
             pass 
             
        grid_sig = sig_norm.view(side, side) # (H, W)
        
        # 4. Trajectory to 2D (Position Encoding)
        grid_traj = traj.view(side, side, 3) # (H, W, 3)
        # Normalize traj to [-1, 1] approx (bounds are -0.2 to 0.2)
        grid_traj = grid_traj / 0.2 
        
        # 5. Stack into (4, H, W) [Signal, X, Y, Z]
        # Signal: (1, H, W)
        # Traj: (3, H, W) -> permute
        img_input = torch.cat([
            grid_sig.unsqueeze(0),
            grid_traj.permute(2, 0, 1)
        ], dim=0).float()
        
        sample = {
            "signals": img_input, # Now (4, H, W)
            "mask": tensors["mask"][inner_idx],
            "traj": traj, # Keep original traj just in case
        }
        if self.load_bfield and "bfield" in tensors:
            sample["bfield"] = tensors["bfield"][inner_idx]
            
        return sample
