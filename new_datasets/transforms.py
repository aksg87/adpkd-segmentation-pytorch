from torchvision import transforms
from PIL import Image

import torch

BACKGROUND = 0.0
L_KIDNEY = 0.5019608
R_KIDNEY = 0.7490196


class BaselineMaskEncode:
    """Default mask2label function.

    The first channel for right kidney vs background,
    and the second one for left kidney vs background.

    Kidneys are marked as 1, background as 0.

    Args:
        mask, (1, H, W) float32 tensor

    Returns:
        tensor, (2, H, W) uint8 one-hot encoded label
    """

    def __call__(self, mask):
        r_kidney = (mask == R_KIDNEY).type(torch.uint8)
        l_kidney = (mask == L_KIDNEY).type(torch.uint8)
        return torch.cat([r_kidney, l_kidney], dim=0)


class SingleChannelMask:
    """Sets 1 for kidneys, 0 otherwise.

    Args:
        mask, (1, H, W) float32 tensor

    Returns:
        tensor, (1, H, W) uint8 one-hot encoded label
    """
    def __call__(self, mask):
        kidney = torch.bitwise_or(mask == R_KIDNEY, mask == L_KIDNEY)
        return kidney.type(torch.uint8)


class ThreeChannelMask:
    """One channel for each of the 3 classes.

    Args:
        mask, (1, H, W) float32 tensor

    Returns:
        tensor, (3, H, W) uint8 one-hot encoded label
    """
    def __call__(self, mask):
        background = (mask == BACKGROUND).type(torch.uint8)
        r_kidney = (mask == R_KIDNEY).type(torch.uint8)
        l_kidney = (mask == L_KIDNEY).type(torch.uint8)
        return torch.cat([r_kidney, l_kidney, background], dim=0)


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
