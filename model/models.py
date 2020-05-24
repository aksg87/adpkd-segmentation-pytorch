# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import segmentation_models_pytorch as smp

from model.model_utils import params

# %%

model = smp.Unet(**params)     
