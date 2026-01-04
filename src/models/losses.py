from typing import Dict

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum(dim=list(range(1, probs.dim())))
    den = (probs + target).sum(dim=list(range(1, probs.dim()))) + eps
    return (1 - num / den).mean()  # Average over batch to get scalar


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss for class imbalance.
    alpha: weight for positive class (higher = more focus on positives)
    gamma: focusing parameter (higher = more focus on hard examples)
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    probs = torch.sigmoid(logits)
    pt = torch.where(target == 1, probs, 1 - probs)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    focal_weight = alpha_t * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6) -> torch.Tensor:
    """
    Tversky Loss - generalization of Dice that controls FP/FN balance.
    alpha: weight for false positives (lower = less penalty for FP)
    beta: weight for false negatives (higher = more penalty for FN, finds more positives)
    For class imbalance: use alpha < 0.5, beta > 0.5 to encourage finding rare positives.
    """
    probs = torch.sigmoid(logits)
    tp = (probs * target).sum(dim=list(range(1, probs.dim())))
    fp = (probs * (1 - target)).sum(dim=list(range(1, probs.dim())))
    fn = ((1 - probs) * target).sum(dim=list(range(1, probs.dim())))
    tversky = tp / (tp + alpha * fp + beta * fn + eps)
    return (1 - tversky).mean()


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return dice_loss(logits, target) + bce_loss(logits, target)


def dice_focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combination of Dice Loss + Focal Loss for extreme class imbalance."""
    return dice_loss(logits, target) + focal_loss(logits, target)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta)


class LossComputer:
    def __init__(self, mask_weight: float = 1.0, b_weight: float = 0.2, pos_weight: float = 100.0):
        """
        Loss computer for segmentation.
        
        Args:
            mask_weight: Weight for mask loss
            b_weight: Weight for B-field loss
            pos_weight: Weight for positive class in BCE (higher = more focus on rare positives)
        """
        self.mask_weight = mask_weight
        self.b_weight = b_weight
        self.pos_weight = pos_weight
        self._pos_weight_tensor = None

    def __call__(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        if "mask_logits" in outputs:
            # Create pos_weight tensor on correct device (lazy init)
            if self._pos_weight_tensor is None or self._pos_weight_tensor.device != outputs["mask_logits"].device:
                self._pos_weight_tensor = torch.tensor([self.pos_weight], device=outputs["mask_logits"].device)
            
            # Combo Loss: Weighted BCE + Dice (proven effective in overfitting test)
            # BCE provides stable pixel-wise gradients
            # Dice directly optimizes overlap, ignoring class imbalance
            bce = F.binary_cross_entropy_with_logits(
                outputs["mask_logits"], 
                batch["mask"],
                pos_weight=self._pos_weight_tensor
            )
            dice = dice_loss(outputs["mask_logits"], batch["mask"])
            
            # 0.5 * BCE + 0.5 * Dice (balanced combo)
            losses["mask"] = self.mask_weight * (0.5 * bce + 0.5 * dice)
            
        if self.b_weight > 0 and "bfield" in outputs and "bfield" in batch:
            losses["bfield"] = self.b_weight * huber_loss(outputs["bfield"], batch["bfield"])
        total = sum(losses.values()) if losses else torch.tensor(0.0, device=outputs["mask_logits"].device)
        losses["total"] = total
        return losses

