"""Dataset definitions"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import data_set.datasets as ds

class BaselineDatasetGetter(torch.utils.data.Dataset):
    def __init__(self, dataset_type, dataset_params):
        self.dataset_type = dataset_type
        self.dataset_params = dataset_params

    def __call__(self):
        dataset = getattr(ds, self.dataset_type)

        return dataset(**self.dataset_params)