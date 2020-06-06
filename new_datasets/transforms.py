from torchvision import transforms
from PIL import Image

import torch

L_KIDNEY = 0.5019608
R_KIDNEY = 0.7490196


class BaselineMaskEncode:
    """
    Default mask2label function.

    The first channel for right kidney vs background,
    and the second one for left kidney vs background.

    Kidneys are marked as 1, background as 0.

    Args:
        mask, (1, H, W) float32 tensor

    Returns:
        tensor, (2, H, W) uint8 one-hot encoded label
    """
    def __call__(self, mask):
        unique_vals = [R_KIDNEY, L_KIDNEY]
        mask = mask.squeeze()
        s = mask.shape
        ones = torch.ones(s, dtype=torch.uint8)
        zeros = torch.zeros(s, dtype=torch.uint8)
        one_hot_map = [
            torch.where(mask == unique_vals[targ], ones, zeros)
            for targ in range(len(unique_vals))
        ]
        one_hot_map = torch.stack(one_hot_map, dim=0)

        return one_hot_map


class SingleChannelMask:
    def __call__():
        pass


class Transform_X:
    def __init__(self, dim=96):
        self.T_x = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((dim, dim), interpolation=Image.CUBIC),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(x.shape).expand(3, -1, -1)),
            ]
        )

    def __call__(self):
        return self.T_x


class Transform_Y:
    def __init__(self, dim=96, mask2label=None):
        if mask2label is None:
            mask2label = BaselineMaskEncode()
        self.mask2label = mask2label

        self.T_y = transforms.Compose(
            [
                transforms.ToPILImage(),
                # "non-nearest" interpolation breaks mask --> one-hot-encode
                transforms.Resize((dim, dim), interpolation=Image.NEAREST),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: self.mask2label(x)),
            ]
        )

    def __call__(self):
        return self.T_y
