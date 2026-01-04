from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml


def save_metadata(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_metadata(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_shard(tensors: Dict[str, torch.Tensor], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, path)


def load_shard(path: str | Path) -> Dict[str, torch.Tensor]:
    path = Path(path)
    # mmap was added in newer PyTorch; fall back gracefully if unsupported.
    try:
        return torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")
    except RuntimeError as exc:
        if "mmap" in str(exc).lower():
            return torch.load(str(path), map_location="cpu")
        raise


def shard_name(split: str, idx: int) -> str:
    return f"{split}_{idx:05d}.pt"


def split_counts(total: int, shard_size: int) -> Tuple[int, int]:
    shards = total // shard_size
    remainder = total % shard_size
    return shards, remainder
