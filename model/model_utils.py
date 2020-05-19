
import torch
import numpy as np

import segmentation_models_pytorch as smp
import albumentations as albu

# %%
params = {
    "encoder_name": 'efficientnet-b6',
    "encoder_weights": 'imagenet',
    "activation": None,
    "classes": 2
}

# %%
def get_tensor(x, **kwargs):
    """helper function for get_preprocessing"""
    x = np.array(x)
    return x.transpose(1, 2, 0).astype('float32')

def to_tensor(x, **kwargs):
    """helper function for get_preprocessing"""
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Preprocessing  functions for segmentation-models-pytorch pretrained models
    https://github.com/qubvel/segmentation_models.pytorch
    """    
    _transform = [
        albu.Lambda(image=get_tensor, mask=get_tensor),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

preprocessing_fn = smp.encoders.get_preprocessing_fn(params["encoder_name"], params["encoder_weights"])


def load_model(model, name):
    model.load_state_dict(torch.load("../saved/models/{}".format(name)))
    return model