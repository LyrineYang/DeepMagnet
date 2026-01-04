from dataclasses import dataclass
from typing import Tuple

import torch

from .coil import CoilConfig, coil_moments

MU0 = 4e-7 * torch.pi


@dataclass
class GridConfig:
    size: Tuple[int, int, int]
    bounds: Tuple[float, float]
    voxel_size: float


def dipole_field(centers: torch.Tensor, moments: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Compute magnetic field of dipoles at query points.
    centers: (n,3), moments: (n,3), points: (...,3)
    returns: (...,3)
    """
    points_exp = points.unsqueeze(-2)  # (...,1,3)
    r = points_exp - centers  # (...,n,3)
    r_norm = torch.norm(r, dim=-1, keepdim=True) + 1e-8
    m_dot_r = torch.sum(moments * r, dim=-1, keepdim=True)  # (...,n,1)
    term1 = 3 * m_dot_r * r / (r_norm ** 5)
    term2 = moments / (r_norm ** 3)
    b = MU0 / (4 * torch.pi) * (term1 - term2)
    return b.sum(dim=-2)


def coil_field_at_points(cfg: CoilConfig, traj: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Compute field at query points for each trajectory position."""
    centers_moments = coil_moments(cfg, device=traj.device)
    centers = centers_moments[:, :3]  # (n, 3)
    moments = centers_moments[:, 3:]  # (n, 3)
    n_coils = centers.shape[0]
    steps = traj.shape[0]
    
    # For each traj point, shift centers accordingly.
    shifted_centers = centers.unsqueeze(0) + traj.unsqueeze(1)  # (steps, n, 3)
    
    # Handle points: can be (3,) for single point or (steps, 3) for per-step points
    if points.dim() == 1:
        points = points.unsqueeze(0).expand(steps, -1)  # (steps, 3)
    points_exp = points.unsqueeze(1)  # (steps, 1, 3)
    
    r = points_exp - shifted_centers  # (steps, n, 3)
    r_norm = torch.norm(r, dim=-1, keepdim=True) + 1e-8
    m_dot_r = torch.sum(moments.unsqueeze(0) * r, dim=-1, keepdim=True)
    term1 = 3 * m_dot_r * r / (r_norm ** 5)
    term2 = moments.unsqueeze(0) / (r_norm ** 3)
    b = MU0 / (4 * torch.pi) * (term1 - term2)
    return b.sum(dim=1)  # (steps, 3)


def synthesize_signal(
    cfg: CoilConfig,
    traj: torch.Tensor,
    mask: torch.Tensor,
    grid_coords: torch.Tensor,
    noise_std: float | None = None,
) -> torch.Tensor:
    """
    Generate synthetic waveform for each trajectory step.
    mask: (Dx,Dy,Dz) float, grid_coords: (Dx,Dy,Dz,3)
    returns: (steps, samples)
    """
    noise_std = cfg.noise_std if noise_std is None else noise_std
    device = traj.device
    # Compute object centroid as proxy for coupling strength.
    mass = mask.sum()
    if mass > 0:
        centroid = (mask.unsqueeze(-1) * grid_coords).view(-1, 3).sum(dim=0) / mass
    else:
        centroid = torch.zeros(3, device=device)

    b_at_centroid = coil_field_at_points(cfg, traj, centroid)
    amplitude = torch.norm(b_at_centroid, dim=-1) * torch.clamp(mass, min=1.0) * 1e6
    # Distance attenuation: further objects produce weaker signals.
    dist = torch.norm(traj - centroid, dim=-1)
    attenuation = torch.exp(-dist / 0.1)
    amplitude = amplitude * attenuation

    t = torch.linspace(0, 1, cfg.samples, device=device)
    wave = torch.sin(2 * torch.pi * t * cfg.frequency)
    signal = amplitude.unsqueeze(-1) * wave  # (steps,samples)
    if noise_std > 0:
        signal = signal + noise_std * torch.randn_like(signal)
    return signal


def field_volume(cfg: CoilConfig, traj_idx: int, traj: torch.Tensor, grid_coords: torch.Tensor) -> torch.Tensor:
    """
    Compute coil field on a grid for visualization at a single trajectory index.
    grid_coords: (Dx,Dy,Dz,3)
    returns: (Dx,Dy,Dz,3)
    """
    centers_moments = coil_moments(cfg, device=traj.device)
    centers = centers_moments[:, :3] + traj[traj_idx].unsqueeze(0)
    moments = centers_moments[:, 3:]
    points = grid_coords.view(-1, 3)
    b = dipole_field(centers, moments, points)
    return b.view(*grid_coords.shape[:-1], 3)


def add_realistic_noise(signal: torch.Tensor, noise_level: float, drift_weight: float = 0.2) -> torch.Tensor:
    """
    Inject realistic sensor noise combining white noise and low-frequency drift.
    
    Args:
        signal: Input signal tensor.
        noise_level: Magnitude of noise (0.0 to 1.0 references relative scale).
        drift_weight: Proportion of noise that is low-frequency drift.
    
    Returns:
        Noisy signal.
    """
    if noise_level <= 0:
        return signal
        
    # 1. White Gaussian Noise
    white_noise = torch.randn_like(signal)
    
    # 2. Low-frequency Drift (Random Walk / Cumulative Sum)
    # We generate a random walk and smooth it or just use cumsum of smaller noise
    drift = torch.cumsum(torch.randn_like(signal), dim=-1)
    # Normalize drift to have similar scale to white noise before weighting
    drift = drift / (torch.std(drift) + 1e-8)
    
    # Combine
    mixed_noise = (1 - drift_weight) * white_noise + drift_weight * drift
    
    # Scale by noise level and signal amplitude (or fixed scale if signal is normalized)
    # Assuming signal roughly range [-1, 1] or similar order of magnitude
    # We'll use a fixed scale factor heuristic or relative to signal std
    scale = noise_level # 0.5 is "high" noise in this context usually
    
    noisy_signal = signal + scale * mixed_noise
    return noisy_signal
