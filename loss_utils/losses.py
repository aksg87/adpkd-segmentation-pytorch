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
def soft_dice_loss(y_pred, y_true, epsilon=1e-6, X_Y=(2, 3)):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_first` format by default: X_Y = (2, 3)

    # Arguments

        y_pred: b x c x (X x Y) Network output, must sum to 1 over
        y_true: b x c x (X x Y) One hot encoding of ground truth
        c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors

    # Reference in Numpy
        https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08
    """

    numerator = 2.0 * torch.sum(y_pred * y_true, X_Y)
    denominator = torch.sum(torch.pow(y_pred, 2) + torch.pow(y_true, 2), X_Y)

    return 1 - torch.mean(
        numerator / (denominator + epsilon)
    )  # average over classes and batch


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
