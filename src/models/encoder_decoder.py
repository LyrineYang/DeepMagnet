from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Decoder3DConfig:
    base_channels: int = 32
    depth: int = 4


def conv3d_block(in_ch: int, out_ch: int, k: int = 3) -> nn.Sequential:
    pad = k // 2
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=pad),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class SignalEncoder(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, steps, samples)
        b, steps, samples = x.shape
        x = x.view(b, 1, steps * samples)
        return self.net(x).squeeze(-1)  # (B,H)


class VolumeDecoder(nn.Module):
    def __init__(self, cfg: Decoder3DConfig, output_channels: int = 1):
        super().__init__()
        ch = cfg.base_channels
        layers = []
        in_ch = ch
        for _ in range(cfg.depth):
            layers.append(nn.ConvTranspose3d(in_ch, ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm3d(ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = ch
            ch = max(ch // 2, 8)
        self.net = nn.Sequential(*layers)
        self.mask_head = nn.Conv3d(in_ch, 1, kernel_size=1)
        self.b_head = nn.Conv3d(in_ch, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.net(x)
        return {"mask_logits": self.mask_head(feat), "bfield": self.b_head(feat)}


class SeqToVol(nn.Module):
    def __init__(self, latent_dim: int = 256, decoder_cfg: Decoder3DConfig = Decoder3DConfig(), grid_shape=(64, 64, 64)):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_cfg = decoder_cfg
        self.encoder = SignalEncoder(hidden=latent_dim)
        # Start from a small spatial seed; adjust to reach grid size.
        self.seed_shape = (4, 4, 4)
        seed_voxels = self.seed_shape[0] * self.seed_shape[1] * self.seed_shape[2]
        self.fc = nn.Linear(latent_dim, decoder_cfg.base_channels * seed_voxels)
        self.decoder = VolumeDecoder(decoder_cfg)
        self.grid_shape = grid_shape

    def forward(self, signals: torch.Tensor) -> dict:
        b = signals.shape[0]
        latent = self.encoder(signals)
        seed = self.fc(latent).view(b, self.decoder_cfg.base_channels, *self.seed_shape)
        out = self.decoder(seed)
        # Optionally resize to target grid via interpolation.
        mask = torch.nn.functional.interpolate(out["mask_logits"], size=self.grid_shape, mode="trilinear", align_corners=False)
        bfield = torch.nn.functional.interpolate(out["bfield"], size=self.grid_shape, mode="trilinear", align_corners=False)
        return {"mask_logits": mask.squeeze(1), "bfield": bfield}
