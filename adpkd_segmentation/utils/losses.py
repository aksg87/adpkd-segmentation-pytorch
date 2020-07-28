"""Loss utilities and definitions"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from adpkd_segmentation.data.data_utils import (
    KIDNEY_PIXELS,
    STUDY_TKV,
    VOXEL_VOLUME,
)


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


class Dice(nn.Module):
    """Dice metric/loss.

    Supports different Dice variants.
    """

    def __init__(
        self,
        pred_process,
        epsilon=1e-8,
        power=2,
        dim=(2, 3),
        standardize_func=None,
        use_as_loss=True,
    ):
        super().__init__()
        self.pred_process = pred_process
        self.epsilon = epsilon
        self.power = power
        self.dim = dim
        self.standardize_func = standardize_func
        self.use_as_loss = use_as_loss

    def __call__(self, pred, target):
        pred = self.pred_process(pred)
        if self.standardize_func is not None:
            pred = self.standardize_func(pred)
            target = self.standardize_func(target)

        intersection = torch.sum(pred * target, dim=self.dim)
        set_add = torch.sum(
            pred ** self.power + target ** self.power, dim=self.dim
        )
        score = (2 * intersection + self.epsilon) / (set_add + self.epsilon)
        score = score.mean()
        if self.use_as_loss:
            return 1 - score
        return score


class KidneyPixelMAPE(nn.Module):
    """
    Calculates the absolute percentage error for predicted kidney pixel counts

    (label kidney pixel count - predicted k.p. count) / (label k.p. count)

    By default, kidney pixel summation is done for each image separately, and
    averaged over the entire batch.

    Depending on the `pred_process` function,
    predicted kidney pixel count can be soft or hard.
    """

    def __init__(
        self, pred_process, epsilon=1.0, dim=(2, 3), standardize_func=None
    ):
        super().__init__()
        self.pred_process = pred_process
        self.epsilon = epsilon
        self.dim = dim
        self.standardize_func = standardize_func

    def __call__(self, pred, target):

        pred = self.pred_process(pred)
        if self.standardize_func is not None:
            pred = self.standardize_func(pred)
            target = self.standardize_func(target)

        target_count = target.sum(dim=self.dim).detach()
        pred_count = pred.sum(dim=self.dim)

        kp_batch_MAPE = torch.abs(
            (target_count - pred_count) / (target_count + self.epsilon)
        ).mean()

        return kp_batch_MAPE


class KidneyPixelMSLE(nn.Module):
    """
    Mean square error for the log of kidney pixel counts.

    MSE of ln(label kidney pixel count) - ln(predicted k.p. count)

    By default, pixels are counted separetely for each image, with final
    averaging across all images

    Depending on the `pred_process` function,
    predicted kidney pixel count can be soft or hard.
    """

    def __init__(
        self, pred_process, epsilon=1.0, dim=(2, 3), standardize_func=None
    ):
        super().__init__()
        self.pred_process = pred_process
        self.epsilon = epsilon
        self.dim = dim
        self.standardize_func = standardize_func

    def __call__(self, pred, target):
        pred = self.pred_process(pred)
        if self.standardize_func is not None:
            pred = self.standardize_func(pred)
            target = self.standardize_func(target)

        target_count = target.sum(dim=self.dim).detach()
        pred_count = pred.sum(dim=self.dim)

        sle = (
            torch.log(target_count + self.epsilon)
            - torch.log(pred_count + self.epsilon)
        ) ** 2
        msle = torch.mean(sle)
        return msle


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
    def __init__(
        self, criterions, epsilon=1e-6, weights=None, requires_extra_dict=None
    ):
        self.criterions = criterions
        self.epsilon = epsilon
        self.requires_extra_dict = requires_extra_dict
        self.weights = weights
        if weights is None:
            self.weights = [1.0] * len(self.criterions)
        self.weights = torch.tensor(self.weights)
        if requires_extra_dict is None:
            self.requires_extra_dict = [False for c in self.criterions]

    def __call__(self, pred, target, extra_dict=None):
        # first, scale losses such that
        # L_1 * s_1 = L_2 * s_2 = ... L_n * s_n =
        # L_1 * L_2 * ... * L_n
        # e.g. s_2 = L_1 * L_3 * ... * L_n
        # calculate scaling factors dynamically
        partial_losses = []
        for c, e in zip(self.criterions, self.requires_extra_dict):
            loss = c(pred, target, extra_dict) if e else c(pred, target)
            partial_losses.append(loss)
        partial_losses = torch.stack(partial_losses) + self.epsilon
        # no backprop through dynamic scaling factors
        detached = partial_losses.detach()
        prod = torch.prod(detached)
        # divide the total product by the vector of loss values
        # to get scaling factors such as e.g. s_2 = L_1 * L_3 * ... * L_n
        scales = prod / detached
        # final weighting by external weights
        self.weights = self.weights.to(scales.device)
        scales = scales * self.weights
        normalization = torch.sum(scales)

        loss = (partial_losses * scales).sum() / normalization
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


class BiasReductionLoss(nn.Module):
    def __init__(
        self, pred_process, standardize_func=None, w1=0.5, w2=0.5, epsilon=1e-8
    ):
        super().__init__()
        self.pred_process = pred_process
        self.standardize_func = standardize_func
        self.w1 = w1
        self.w2 = w2
        self.epsilon = epsilon

    def __call__(self, pred, target):
        pred = self.pred_process(pred)
        if self.standardize_func is not None:
            pred = self.standardize_func(pred)
            target = self.standardize_func(target)

        intersection = torch.sum(pred * target, dim=(1, 2, 3))
        # count what's missing from the target area
        missing = target.sum(dim=(1, 2, 3)) - intersection
        # count all extra predictions outside the target area
        false_pos = torch.sum(pred * (1 - target), dim=(1, 2, 3))

        # both losses should go to zero, but they should also be the same
        loss = (
            self.w1 * (missing ** 2 + false_pos ** 2)
            + self.w2 * (missing - false_pos) ** 2
        )
        # sqrt is not differentiable at zero
        loss = (loss.mean() + self.epsilon) ** 0.5

        return loss
