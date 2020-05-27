import numpy as np

import segmentation_models_pytorch as smp
import albumentations as albu


def get_tensor(x, **kwargs):
    """helper function for get_preprocessing"""
    x = np.array(x)
    return x.transpose(1, 2, 0).astype("float32")


def to_tensor(x, **kwargs):
    """helper function for get_preprocessing"""


class BaselinePreprocessGetter:
    """preprocessing for smp module and data prep"""

    def __init__(self, encoder_name, encoder_weights):

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            encoder_name, encoder_weights
        )

        self._transform = [
            albu.Lambda(image=get_tensor, mask=get_tensor),
            albu.Lambda(image=self.preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]

    def __call__(self):

        return albu.Compose(self._transform)
