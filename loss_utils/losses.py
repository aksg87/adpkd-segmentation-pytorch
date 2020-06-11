"""Loss utilities and definitions"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
def binarize_thresholds(pred, thresholds):
    """
    Args:
        pred: model pred tensor with shape b x c x (X x Y)
        thresholds: list of floats i.e. [0.6,0.5,0.4]

    Returns:
        float tensor: binary values
    """

    C = len(thresholds)
    thresholds = torch.tensor(thresholds)
    thresholds = thresholds.reshape(1, C, 1, 1)
    thresholds.expand_as(pred)
    thresholds = thresholds.to(pred.device)
    res = pred > thresholds

    return res.float()


# %%
def binarize_argmax(pred):
    """
    Args:
        pred: model pred tensor with shape b x c x (X x Y)

    Returns:
        float tensor: binary values
    """
    max_c = torch.argmax(pred, 1)  # argmax across C axis
    num_classes = pred.shape[1]
    encoded = torch.nn.functional.one_hot(max_c, num_classes)
    encoded = encoded.permute([0, 3, 1, 2])

    return encoded.float()


def calculate_DSC(y_pred, y_true, epsilon=1e-6, X_Y=(2, 3), power=2):
    """
    Dice simmilarity coeficient calculation.

    Assumes the `channels_first` format by default: X_Y = (2, 3)
    Support binary and soft predictions.

    Separate Dice calculation for each of the class channels, and then
    averaged over all classes and examples in the batch.

    Numpy reference
    https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08

    Args:
        y_pred: b x c x (X x Y) Prediction (sigmoid or binarized)
        y_true: b x c x (X x Y) One hot encoding of ground truth
        epsilon: used for numerical stability to avoid divide by zero errors
        X_Y: tuple, height and width dimensions
        power: 1 or 2, supporting different Dice loss implementations
    """

    numerator = 2.0 * torch.sum(y_pred * y_true, X_Y)
    denominator = torch.sum(
        torch.pow(y_pred, power) + torch.pow(y_true, power), X_Y
    )

    # average over classes and batch
    return torch.mean((epsilon + numerator) / (denominator + epsilon))


class SigmoidBinarize:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, pred):
        # Expects (N, C, H, W) format
        return binarize_thresholds(torch.sigmoid(pred), self.thresholds)


class SigmoidForwardBinarize:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, pred):
        # Expects (N, C, H, W) format
        soft = torch.sigmoid(pred)
        hard = binarize_thresholds(soft, self.thresholds)
        return hard.detach() + soft - soft.detach()


class SoftmaxBinarize:
    def __call__(self, pred):
        # Expects (N, C, H, W) format
        return binarize_argmax(pred)


class SoftmaxForwardBinarize:
    def __call__(self, pred):
        # Expects (N, C, H, W) format
        soft = F.softmax(pred, dim=1)
        hard = binarize_argmax(soft)
        return hard.detach() + soft - soft.detach()


class SoftDice(nn.Module):
    """"New soft dice loss criterion callable"""

    def __init__(self, pred_process, epsilon=1e-6, power=2):
        super().__init__()
        self.pred_process = pred_process
        self.epsilon = epsilon
        self.power = power

    def __call__(self, pred, target):

        pred = self.pred_process(pred)
        loss = 1 - calculate_DSC(
            pred, target, epsilon=self.epsilon, power=self.power
        )

        return loss


class HardDice(nn.Module):
    """Binarized dice in forward pass, soft in backward"""

    def __init__(self, binary_forward, epsilon=1e-6, power=2):
        super().__init__()
        self.binary_forward = binary_forward
        self.epsilon = epsilon
        self.power = power

    def __call__(self, pred, target):

        pred = self.binary_forward(pred)
        dsc = calculate_DSC(
            pred, target, epsilon=self.epsilon, power=self.power
        )

        return 1 - dsc


class DiceMetric(nn.Module):
    """Dice metric callable"""

    def __init__(self, binarize_func, epsilon=1e-6, power=2):
        super().__init__()
        self.binarize_func = binarize_func
        self.epsilon = epsilon
        self.power = power

    def __call__(self, pred, target):

        pred = self.binarize_func(pred)
        dsc = calculate_DSC(
            pred, target, epsilon=self.epsilon, power=self.power
        )

        return dsc


class CombinedDiceBCE(nn.Module):
    """Combined soft Dice and BCE loss"""

    def __init__(self, epsilon=1e-6, bce_weight=0.5, power=2):
        super().__init__()
        self.epsilon = epsilon
        self.bce_weight = bce_weight
        self.power = power
        self.soft_dice = SoftDice(
            pred_process=torch.sigmoid, epsilon=epsilon, power=power
        )

    def __call__(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = self.soft_dice(pred, target)
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)
        return loss


class WeightedLosses(nn.Module):
    def __init__(self, criterions, weights):
        super().__init__()
        self.criterions = criterions
        self.weights = weights

    def __call__(self, pred, target):
        losses = [
            c(pred, target) * w for c, w in zip(self.criterions, self.weights)
        ]
        return torch.sum(torch.stack(losses))


# %%
# old implementation
def dice_loss(pred, target, smooth=1e-8):
    # flatten label and prediction tensors
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


# %%
# old implementation
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
