import torch
import torch.nn as nn


class LinearBaseline(nn.Module):
    def __init__(self, input_length: int, output_voxels: int):
        super().__init__()
        self.fc = nn.Linear(input_length, output_voxels)

    def forward(self, signals: torch.Tensor, grid_shape) -> dict:
        b = signals.size(0)
        x = signals.view(b, -1)
        logits = self.fc(x).view(b, *grid_shape)
        return {"mask_logits": logits, "bfield": torch.zeros(b, 3, *grid_shape, device=signals.device)}
