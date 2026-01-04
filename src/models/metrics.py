from typing import Dict

import torch


def dice_coef(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred = (probs > thresh).float()
    num = 2 * (pred * target).sum(dim=list(range(1, pred.dim())))
    den = (pred + target).sum(dim=list(range(1, pred.dim()))) + eps
    return num / den


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=list(range(1, pred.dim())))
    return 20 * torch.log10(max_val / torch.sqrt(mse + eps))


def compute_metrics(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], thresh: float = 0.5) -> Dict[str, torch.Tensor]:
    metrics = {}
    if "mask_logits" in outputs and "mask" in batch:
        metrics["dice"] = dice_coef(outputs["mask_logits"], batch["mask"], thresh=thresh).mean()
    if "bfield" in outputs and "bfield" in batch:
        metrics["psnr_b"] = psnr(outputs["bfield"], batch["bfield"]).mean()
    return metrics
