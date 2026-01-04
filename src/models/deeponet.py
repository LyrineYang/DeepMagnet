from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, k: int, stride: int = 1) -> nn.Sequential:
    pad = k // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


def mlp_block(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True))


@dataclass
class BranchConfig:
    hidden: int = 128
    layers: int = 4
    kernel_size: int = 5
    type: str = "cnn1d"  # or transformer placeholder


@dataclass
class TrunkConfig:
    hidden: int = 256
    layers: int = 6


class BranchNet(nn.Module):
    def __init__(self, cfg: BranchConfig, input_length: int):
        super().__init__()
        layers = []
        in_ch = 1
        channels = cfg.hidden
        current = input_length
        for _ in range(cfg.layers):
            layers.append(conv_block(in_ch, channels, cfg.kernel_size))
            in_ch = channels
            current = max(1, current // 2)
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.net = nn.Sequential(*layers, nn.Flatten())
        self.head = nn.Linear(in_ch * current, cfg.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, steps, samples) -> flatten time dimension for 1D conv
        b, steps, samples = x.shape
        x = x.view(b, 1, steps * samples)
        feat = self.net(x)
        return self.head(feat)


class TrunkNet(nn.Module):
    def __init__(self, cfg: TrunkConfig, in_dim: int = 3):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(cfg.layers):
            layers.append(mlp_block(dim, cfg.hidden))
            dim = cfg.hidden
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(dim, cfg.hidden)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (P,3) or (B,P,3)
        orig_shape = coords.shape
        coords = coords.view(-1, orig_shape[-1])
        feat = self.net(coords)
        feat = self.out(feat)
        return feat.view(*orig_shape[:-1], -1)


class DeepONet(nn.Module):
    def __init__(self, branch_cfg: BranchConfig, trunk_cfg: TrunkConfig, out_channels: int = 4):
        super().__init__()
        self.branch_cfg = branch_cfg
        self.trunk_cfg = trunk_cfg
        self.out_channels = out_channels
        self.branch = None  # will be built lazily
        self.trunk = TrunkNet(trunk_cfg)
        
        # Learnable scale factor to amplify weak branch features
        self.branch_scale = nn.Parameter(torch.ones(1) * 10.0)
        
        # Feature normalization before fusion
        self.branch_norm = nn.LayerNorm(branch_cfg.hidden)
        self.trunk_norm = nn.LayerNorm(trunk_cfg.hidden)
        
        # Fusion layer: combine branch and trunk features
        # Use concatenation + MLP instead of pure multiplication
        self.fusion = nn.Sequential(
            nn.Linear(branch_cfg.hidden * 2, branch_cfg.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(branch_cfg.hidden, branch_cfg.hidden),
        )
        
        self.mask_head = nn.Linear(branch_cfg.hidden, 1)
        self.b_head = nn.Linear(branch_cfg.hidden, 3) if out_channels > 1 else None
        
        # Initialize mask_head bias for class imbalance (pi=0.01)
        import math
        pi = 0.01
        self.mask_head.bias.data.fill_(-math.log((1 - pi) / pi))

    def build(self, sample_shape: torch.Size) -> None:
        # sample_shape: (steps, samples)
        steps, samples = sample_shape
        self.branch = BranchNet(self.branch_cfg, input_length=steps * samples)

    def forward(self, signals: torch.Tensor, coords: torch.Tensor) -> dict:
        if self.branch is None:
            self.build(signals.shape[1:])
            # Ensure branch is on same device
            self.branch = self.branch.to(signals.device)
            
        branch_feat = self.branch(signals)  # (B, H)
        trunk_feat = self.trunk(coords)      # (P, H) or (B, P, H)
        
        # Normalize and scale branch features
        branch_feat = self.branch_norm(branch_feat) * self.branch_scale
        
        if trunk_feat.dim() == 2:
            # coords shared across batch: trunk_feat is (P, H)
            trunk_feat = self.trunk_norm(trunk_feat)
            B = branch_feat.size(0)
            P = trunk_feat.size(0)
            
            # Memory-efficient fusion: avoid materializing full (B, P, H) twice
            # Use broadcasting + in-place concat
            branch_exp = branch_feat.unsqueeze(1).expand(B, P, -1)  # (B, P, H)
            trunk_exp = trunk_feat.unsqueeze(0).expand(B, P, -1)    # (B, P, H)
            
            # Concatenate and fuse (single (B, P, H*2) tensor)
            concat = torch.cat([branch_exp, trunk_exp], dim=-1)  # (B, P, H*2)
            features = self.fusion(concat)  # (B, P, H)
            # Note: Removed multiplicative residual to save memory
        else:
            trunk_feat = self.trunk_norm(trunk_feat)
            concat = torch.cat([branch_feat.unsqueeze(1).expand_as(trunk_feat), trunk_feat], dim=-1)
            features = self.fusion(concat)
            
        mask_logits = self.mask_head(features).squeeze(-1)
        outputs = {"mask_logits": mask_logits}
        if self.b_head is not None:
            outputs["bfield"] = self.b_head(features)
        return outputs
