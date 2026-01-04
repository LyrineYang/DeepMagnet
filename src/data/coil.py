from dataclasses import dataclass
from typing import Literal

import torch

TrajectoryType = Literal["line", "grid", "circle"]
CoilType = Literal["double_d", "circle"]


@dataclass
class CoilConfig:
    type: CoilType = "double_d"
    radius: float = 0.05
    separation: float = 0.02
    turns: int = 50
    current: float = 1.5
    frequency: float = 8000.0
    sweep_steps: int = 64
    trajectory: TrajectoryType = "line"
    samples: int = 256
    noise_std: float = 0.01


def sample_trajectory(cfg: CoilConfig, device: torch.device) -> torch.Tensor:
    """Generate coil centers along a chosen trajectory."""
    t = torch.linspace(0, 1, cfg.sweep_steps, device=device)
    if cfg.trajectory == "line":
        # Straight line sweep along x with fixed height.
        x = torch.lerp(torch.tensor(-0.08, device=device), torch.tensor(0.08, device=device), t)
        y = torch.zeros_like(t)
        z = torch.full_like(t, 0.04)
    elif cfg.trajectory == "grid":
        side = int(cfg.sweep_steps ** 0.5)
        xs = torch.linspace(-0.06, 0.06, side, device=device)
        ys = torch.linspace(-0.06, 0.06, side, device=device)
        xv, yv = torch.meshgrid(xs, ys, indexing="ij")
        z = torch.full_like(xv, 0.04)
        traj = torch.stack([xv.flatten(), yv.flatten(), z.flatten()], dim=-1)
        return traj
    elif cfg.trajectory == "circle":
        theta = 2 * torch.pi * t
        x = 0.06 * torch.cos(theta)
        y = 0.06 * torch.sin(theta)
        z = torch.full_like(x, 0.04)
    else:
        raise ValueError(f"Unknown trajectory {cfg.trajectory}")
    return torch.stack([x, y, z], dim=-1)


def coil_moments(cfg: CoilConfig, device: torch.device) -> torch.Tensor:
    """Return magnetic moments for coil loops (double-D as two circles)."""
    area = torch.pi * cfg.radius ** 2
    base_m = cfg.turns * cfg.current * area
    if cfg.type == "double_d":
        offset = torch.tensor([cfg.separation / 2, 0.0, 0.0], device=device)
        m1 = torch.tensor([0.0, 0.0, base_m], device=device)
        m2 = torch.tensor([0.0, 0.0, base_m], device=device)
        centers = torch.stack([-offset, offset], dim=0)
        moments = torch.stack([m1, m2], dim=0)
    elif cfg.type == "circle":
        centers = torch.zeros((1, 3), device=device)
        moments = torch.tensor([[0.0, 0.0, base_m]], device=device)
    else:
        raise ValueError(f"Unknown coil type {cfg.type}")
    return torch.cat([centers, moments], dim=1)  # shape (n,6) center(0:3), moment(3:6)
