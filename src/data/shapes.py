import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from torchvision import datasets, transforms
except ImportError:  # pragma: no cover
    datasets = None
    transforms = None


@dataclass
class MnistConfig:
    root: str = "data/mnist"
    threshold: float = 0.4
    scale_range: Tuple[float, float] = (0.4, 0.8)
    extrude_depth: float = 0.2
    rotate: bool = True
    dilation: int = 0  # Morphological dilation: 0=none, 1=3x3, 2=5x5, etc.


@dataclass
class ShapeConfig:
    types: List[str]
    size_range: Tuple[float, float]
    min_distance_from_coil: float
    weights: Optional[List[float]] = None
    mnist: Optional[MnistConfig] = None


class MnistProvider:
    def __init__(self, cfg: MnistConfig):
        self.cfg = cfg
        self.ds = None

    def _ensure_loaded(self):
        if self.ds is not None:
            return
        if datasets is None or transforms is None:
            raise ImportError("torchvision is required for MNIST shapes. Please install torchvision.")
        self.ds = datasets.MNIST(self.cfg.root, train=True, download=True, transform=transforms.ToTensor())

    def sample(self, device: torch.device) -> torch.Tensor:
        self._ensure_loaded()
        idx = torch.randint(0, len(self.ds), (1,)).item()
        img, _ = self.ds[idx]
        return img.squeeze(0).to(device)  # (H,W) float in [0,1]


def create_grid(size: Tuple[int, int, int], bounds: Tuple[float, float], device: torch.device) -> torch.Tensor:
    """Return grid coordinates tensor of shape (Dx,Dy,Dz,3)."""
    x = torch.linspace(bounds[0], bounds[1], size[0], device=device)
    y = torch.linspace(bounds[0], bounds[1], size[1], device=device)
    z = torch.linspace(bounds[0], bounds[1], size[2], device=device)
    gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1)


def box_mask(grid: torch.Tensor, center: torch.Tensor, size: float) -> torch.Tensor:
    half = size / 2
    cond = (grid >= (center - half)) & (grid <= (center + half))
    return cond.all(dim=-1).float()


def cylinder_mask(grid: torch.Tensor, center: torch.Tensor, radius: float, height: float) -> torch.Tensor:
    xy = grid[..., :2] - center[:2]
    z = grid[..., 2] - center[2]
    r2 = (xy ** 2).sum(dim=-1)
    return ((r2 <= radius ** 2) & (z.abs() <= height / 2)).float()


def compound_mask(grid: torch.Tensor, center: torch.Tensor, size: float) -> torch.Tensor:
    # Union of box and cylinder for richer shapes.
    box = box_mask(grid, center, size)
    cyl = cylinder_mask(grid, center + torch.tensor([size * 0.2, 0.0, 0.0], device=grid.device), size * 0.5, size)
    return torch.clamp(box + cyl, max=1.0)


def random_center(bounds: Tuple[float, float], margin: float, device: torch.device) -> torch.Tensor:
    low = bounds[0] + margin
    high = bounds[1] - margin
    coords = [random.uniform(low, high) for _ in range(3)]
    return torch.tensor(coords, device=device)


def mnist_volume(grid: torch.Tensor, cfg: ShapeConfig, provider: MnistProvider) -> Tuple[torch.Tensor, dict]:
    device = grid.device
    mn_cfg = cfg.mnist
    if mn_cfg is None:
        raise ValueError("MNIST config is missing.")
    img = provider.sample(device)
    img = (img > mn_cfg.threshold).float()
    
    # Apply morphological dilation to thicken strokes
    if mn_cfg.dilation > 0:
        kernel_size = 2 * mn_cfg.dilation + 1  # 1->3x3, 2->5x5, etc.
        img = F.max_pool2d(img.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=mn_cfg.dilation).squeeze()
    
    target_hw = max(4, int(grid.shape[0] * random.uniform(*mn_cfg.scale_range)))
    img = F.interpolate(img.view(1, 1, *img.shape), size=(target_hw, target_hw), mode="bilinear", align_corners=False).squeeze()
    if mn_cfg.rotate:
        k = random.randint(0, 3)
        img = torch.rot90(img, k, dims=(0, 1))
    margin = 2
    max_x = max(margin, grid.shape[0] - target_hw - margin)
    max_y = max(margin, grid.shape[1] - target_hw - margin)
    x0 = random.randint(margin, max_x)
    y0 = random.randint(margin, max_y)
    depth = max(1, int(grid.shape[2] * mn_cfg.extrude_depth))
    z0 = random.randint(margin, max(grid.shape[2] - depth - margin, margin))
    mask = torch.zeros(grid.shape[:-1], device=device)
    digit = img.clamp(0, 1).unsqueeze(-1).expand(target_hw, target_hw, depth)
    x1 = min(x0 + target_hw, mask.shape[0])
    y1 = min(y0 + target_hw, mask.shape[1])
    z1 = min(z0 + depth, mask.shape[2])
    mask[x0:x1, y0:y1, z0:z1] = digit[: x1 - x0, : y1 - y0, : z1 - z0]
    meta = {"type": "mnist", "offset": [x0, y0, z0], "size": [target_hw, target_hw, depth]}
    return mask, meta


def random_shape(grid: torch.Tensor, cfg: ShapeConfig, mnist_provider: MnistProvider | None = None) -> Tuple[torch.Tensor, dict]:
    device = grid.device
    if cfg.weights and len(cfg.weights) == len(cfg.types):
        shape_type = random.choices(cfg.types, weights=cfg.weights, k=1)[0]
    else:
        shape_type = random.choice(cfg.types)
    size = random.uniform(cfg.size_range[0], cfg.size_range[1])
    center = random_center((grid[..., 0].min().item(), grid[..., 0].max().item()), cfg.min_distance_from_coil, device)
    if shape_type == "box":
        mask = box_mask(grid, center, size)
    elif shape_type == "cylinder":
        mask = cylinder_mask(grid, center, size / 2, size)
    elif shape_type == "mnist":
        if mnist_provider is None:
            raise ValueError("MNIST provider not initialized.")
        return mnist_volume(grid, cfg, mnist_provider)
    else:
        mask = compound_mask(grid, center, size)
    meta = {"type": shape_type, "center": center.tolist(), "size": size}
    return mask, meta
