from torchvision import transforms
from PIL import Image

import numpy as np


class BaselineMaskEncode:
    def __call__(self):
        return self.mask2label

    @staticmethod
    def mask2label(mask):
        """converts mask png to one-hot-encoded label"""

        L_KIDNEY = 0.5019608
        R_KIDNEY = 0.7490196
        unique_vals = [R_KIDNEY, L_KIDNEY]

        mask = mask.squeeze()

        s = mask.shape

        ones, zeros = np.ones(s), np.zeros(s)

        one_hot_map = [
            np.where(mask == unique_vals[targ], ones, zeros)
            for targ in range(len(unique_vals))
        ]

        one_hot_map = np.stack(one_hot_map, axis=0).astype(np.uint8)

        return one_hot_map


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
        self.mask2label = mask2label()

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
