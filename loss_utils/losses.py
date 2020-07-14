"""Loss utilities and definitions"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.data_utils import KIDNEY_PIXELS, STUDY_TKV, VOXEL_VOLUME


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


class StandardizeModels:
    def __init__(self, ignore_channel=2):
        # used for backgoround in 3 channel setups
        self.ignore_channel = 2

    def __call__(self, binary_mask):
        # N, C, H, W mask
        num_channels = binary_mask.shape[1]
        if num_channels == 1:
            return binary_mask
        elif num_channels == 2:
            return torch.sum(binary_mask, dim=1, keepdim=True)
        elif num_channels == 3:
            sum_all = torch.sum(binary_mask, dim=1)
            sum_all = sum_all - binary_mask[:, self.ignore_channel, ...]
            sum_all = sum_all.unsqueeze(1)
            return sum_all
        else:
            raise ValueError(
                "Unsupported number of channels: {}".format(num_channels)
            )


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

    def __init__(
        self, binarize_func, epsilon=1e-6, power=2, standardize_func=None
    ):
        super().__init__()
        self.binarize_func = binarize_func
        self.epsilon = epsilon
        self.power = power
        self.standardize_func = standardize_func

    def __call__(self, pred, target):

        pred = self.binarize_func(pred)
        if self.standardize_func is not None:
            pred = self.standardize_func(pred)
            target = self.standardize_func(target)
        dsc = calculate_DSC(
            pred, target, epsilon=self.epsilon, power=self.power
        )

        return dsc


class KidneyPixelMAPE(nn.Module):
    """
    Calculates the absolute percentage error for predicted kidney pixel counts

    (label kidney pixel count - predicted k.p. count) / (label k.p. count)

    The kidney pixel summation is done for each image separately, and
    averaged over the entire batch.

    Depending on the `pred_process` function,
    predicted kidney pixel count can be soft or hard.
    """

    def __init__(self, pred_process, epsilon=1.0, standardize_func=None):
        super().__init__()
        self.pred_process = pred_process
        self.epsilon = epsilon
        self.standardize_func = standardize_func

    def __call__(self, pred, target):

        pred = self.pred_process(pred)
        if self.standardize_func is not None:
            pred = self.standardize_func(pred)
            target = self.standardize_func(target)

        target_count = target.sum(dim=(1, 2, 3)).detach()
        pred_count = pred.sum(dim=(1, 2, 3))

        kp_batch_MAPE = torch.abs(
            (target_count - pred_count) / (target_count + self.epsilon)
        ).mean()

        return kp_batch_MAPE


class KidneyPixelMSLE(nn.Module):
    """
    Mean square error for the log of kidney pixel counts.

    MSE of ln(label kidney pixel count) - ln(predicted k.p. count)

    Pixels are counted separetely for each image, with final averaging
    across all images

    Depending on the `pred_process` function,
    predicted kidney pixel count can be soft or hard.
    """

    def __init__(self, pred_process, epsilon=1.0, standardize_func=None):
        super().__init__()
        self.pred_process = pred_process
        self.epsilon = epsilon
        self.standardize_func = standardize_func

    def __call__(self, pred, target):
        pred = self.pred_process(pred)
        if self.standardize_func is not None:
            pred = self.standardize_func(pred)
            target = self.standardize_func(target)

        target_count = target.sum(dim=(1, 2, 3)).detach()
        pred_count = pred.sum(dim=(1, 2, 3))

        sle = (
            torch.log(target_count + self.epsilon)
            - torch.log(pred_count + self.epsilon)
        ) ** 2
        msle = torch.mean(sle)
        return msle


# deprecated
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
    def __init__(self, criterions, weights, requires_extra_dict=None):
        super().__init__()
        self.criterions = criterions
        self.weights = weights
        self.requires_extra_dict = requires_extra_dict
        if requires_extra_dict is None:
            self.requires_extra_dict = [False for c in self.criterions]

    def __call__(self, pred, target, extra_dict=None):
        losses = []
        for c, w, e in zip(
            self.criterions, self.weights, self.requires_extra_dict
        ):
            loss = c(pred, target, extra_dict) if e else c(pred, target)
            losses.append(loss * w)
        return torch.sum(torch.stack(losses))


class DynamicBalanceLosses(nn.Module):
    def __init__(self, criterions, epsilon=1e-6, requires_extra_dict=None):
        self.criterions = criterions
        self.epsilon = epsilon
        self.requires_extra_dict = requires_extra_dict
        if requires_extra_dict is None:
            self.requires_extra_dict = [False for c in self.criterions]

    def __call__(self, pred, target, extra_dict=None):
        # weights should sum to one (after normalization)
        # L_1 * w_1 = L_2 * w_2 = ... L_n * w_n =
        # L_1 * L_2 * ... * L_n
        # e.g. W_2 = L_1 * L_3 * ... * L_n
        partial_losses = []
        for c, e in zip(self.criterions, self.requires_extra_dict):
            loss = c(pred, target, extra_dict) if e else c(pred, target)
            partial_losses.append(loss)
        partial_losses = torch.stack(partial_losses) + self.epsilon

        # no backprop through weights
        detached = partial_losses.detach()
        prod = torch.prod(detached)
        # divide the total product by the vector of loss values
        # to get weights such as e.g. W_2 = L_1 * L_3 * ... * L_n
        weights = prod / detached
        normalization = torch.sum(weights)

        loss = (partial_losses * weights).sum() / normalization
        return loss


class ErrorLogTKVRelative(nn.Module):
    def __init__(self, pred_process, epsilon=1.0, standardize_func=None):
        super().__init__()
        self.pred_process = pred_process
        self.epsilon = epsilon
        self.standardize_func = standardize_func

    def __call__(self, pred, target, extra_dict):
        pred = self.pred_process(pred)
        if self.standardize_func is not None:
            pred = self.standardize_func(pred)
            target = self.standardize_func(target)

        intersection = torch.sum(pred * target, dim=(1, 2, 3))
        error = (
            torch.sum(pred ** 2, dim=(1, 2, 3))
            + torch.sum(target ** 2, dim=(1, 2, 3))
            - 2 * intersection
        )

        # augmentation correction for original kidney pixel count
        # also, convert to VOXEL VOLUME
        scale = (extra_dict[KIDNEY_PIXELS] + self.epsilon) / (
            torch.sum(target, dim=(1, 2, 3)) + self.epsilon
        )
        scaled_vol_error = scale * error * extra_dict[VOXEL_VOLUME]
        # error more important if kidneys are smaller
        # but for the same kidney volume, error on any slice
        # matters equally
        # use log due to different orders of magnitudes
        weight = 1 / (torch.log(extra_dict[STUDY_TKV]) + self.epsilon)
        log_error = (scaled_vol_error * weight).mean()

        return log_error


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
