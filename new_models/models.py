"""Model definitions"""

import torch.nn as nn
import segmentation_models_pytorch as smp


# the same one as in the current version (smp.Unet)
class BaselineModelGetter(nn.Module):
    def __init__(self, smp_name, smp_params):
        self.smp_name = smp_name
        self.smp_params = smp_params

    def __call__(self):
        smp_model = getattr(smp, self.smp_name)
        return smp_model(**self.smp_params)
