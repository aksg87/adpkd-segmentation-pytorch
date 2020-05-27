# %%
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

import data_set.datasets as ds

# %%
class BaselineDatasetGetter(torch.utils.data.Dataset):
    """Create baseline segmentation dataset"""

    def __init__(self, params):

        self.idxs = params["idxs"]
        self.hyperparam = params["hyperparam"]
        self.transform_x = params["transform_x"]
        self.transform_y = params["transform_y"]
        self.preprocess_func = params["proprocessing_func"]

    def __call__(self):
        
        # allows for passing objects or lists/directionaries for respective variables
        if callable(self.idxs):
            self.idxs = self.idxs()
        
        if callable(self.hyperparam_gen):
            self.hyperparam = self.hyperparam()

        if callable(self.transform_x):
            self.transform_x = self.transform_x()
        
        if callable(self.transform_y):
            self.transform_y = self.transform_y()

        return ds(patient_IDS=self.idxs, hyperparams=self.hyperparam, transform_x = self.transform_x, transform_y =self.transform_x, processing = self.preprocess_func)
        