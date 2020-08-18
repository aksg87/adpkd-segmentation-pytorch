"""Model definitions"""

import torch.nn as nn

from catalyst.dl import utils
import segmentation_models_pytorch as smp


class SmpModelGetter(nn.Module):
    def __init__(self, smp_name, smp_params):
        self.smp_name = smp_name
        self.smp_params = smp_params

    def __call__(self):
        smp_model = getattr(smp, self.smp_name)
        return smp_model(**self.smp_params)


class CatalystModelParamPrep:
    def __init__(self, layerwise_params):
        self.layerwise_params = layerwise_params

    def __call__(self, model):
        return utils.process_model_params(
            model, layerwise_params=self.layerwise_params
        )
