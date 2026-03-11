"""
Loss functions for KransFormer.

  TverskyLoss   — weighted FN/FP penalty
  DiceLoss      — soft Dice
  BCEDiceLoss   — combined BCE + Dice
  CombinedLoss  — main training objective:
                  Tversky + λ_aux·DeepSupervision + λ_reg·KAN-regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """
    Tversky loss for imbalanced segmentation.
    α=0.7 penalises false negatives more — suited for small lesions.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1).float()
        tp = (y_pred * y_true).sum()
        fn = ((1 - y_pred) * y_true).sum()
        fp = (y_pred * (1 - y_true)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1.0 - tversky


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1).float()
        inter  = (y_pred * y_true).sum()
        dice   = (2.0 * inter + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.bce  = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, y_pred, y_true):
        return self.bce_w * self.bce(y_pred, y_true.float()) + self.dice_w * self.dice(y_pred, y_true)


class CombinedLoss(nn.Module):
    """
    L = L_tversky + λ_aux * mean(L_aux_i) + λ_reg * L_kan_reg
    """
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        lambda_aux: float = 0.5,
        lambda_reg: float = 0.1,
        reg_activation: float = 1.0,
        reg_entropy: float = 1.0,
    ):
        super().__init__()
        self.tversky       = TverskyLoss(alpha=alpha, beta=beta)
        self.lambda_aux    = lambda_aux
        self.lambda_reg    = lambda_reg
        self.reg_activation = reg_activation
        self.reg_entropy   = reg_entropy

    def forward(self, outputs, masks: torch.Tensor, model) -> tuple:
        """
        Returns:
            (scalar total loss, main_pred after sigmoid)
        """
        if isinstance(outputs, tuple):
            aux_outputs, main_output = outputs
            main_pred  = torch.sigmoid(main_output)
            loss       = self.tversky(main_pred, masks)
            aux_loss   = sum(self.tversky(torch.sigmoid(a), masks) for a in aux_outputs)
            loss       = loss + self.lambda_aux * aux_loss / len(aux_outputs)
        else:
            main_pred = torch.sigmoid(outputs)
            loss      = self.tversky(main_pred, masks)

        reg  = model.regularization_loss(self.reg_activation, self.reg_entropy)
        loss = loss + self.lambda_reg * reg
        return loss, main_pred
