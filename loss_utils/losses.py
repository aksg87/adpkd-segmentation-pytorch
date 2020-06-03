"""Loss utilities and definitions"""

import torch.nn as nn
import torch.nn.functional as F


# copy from trainer.py as example
def dice_loss(pred, target, smooth=1e-8):
    # flatten label and prediction tensors
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


# current criterion as example
class BaselineLoss(nn.Module):
    "Baseline loss criterion callable"

    def __init__(self, dice_smooth=1e-8, bce_weight=0.5):
        super().__init__()
        self.dice_smooth = dice_smooth
        self.bce_weight = bce_weight

    def __call__(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred = F.sigmoid(pred)
        dice = dice_loss(pred, target, self.dice_smooth)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)

        return loss
