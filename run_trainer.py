# %% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from sklearn.model_selection import train_test_split

from data_set.datasets import SegmentationDataset
from data.data_utils import make_dcmdicts, get_labeled, display_traindata, filter_dcm2attribs
from data.link_data import makelinks

from data_set.datasets import SegmentationDataset
from data_set.transforms import T_x, T_y 
from trainer.trainer import training_loop, criterion
from model.models import model
from model.model_utils import get_preprocessing, preprocessing_fn

# %%
makelinks()
# %%
# find number of patients
_, patient2dcm = make_dcmdicts(tuple(get_labeled()))
all_IDS = range(len(patient2dcm))

# %%
# train, val, test split--> 85 / 15 / 15. Note: Applied to patients not dcm images.
train_IDS, test_IDS = train_test_split(all_IDS, test_size=0.15, random_state=1)
train_IDS, val_IDS = train_test_split(train_IDS, test_size=0.176, random_state=1)

# %%
dataset_train = SegmentationDataset(patient_IDS=train_IDS, transform_x=T_x, transform_y=T_y, preprocessing=get_preprocessing(preprocessing_fn))
dataset_val = SegmentationDataset(patient_IDS=val_IDS, transform_x=T_x, transform_y=T_y, preprocessing=get_preprocessing(preprocessing_fn))

# %%
print("image -> shape {},  dtype {}".format(x.shape, x.dtype))
print("mask -> shape {},  dtype {}".format(y.shape, y.dtype))

# %%
# *** Needs Fix *** - dataset transforms break sample display
# display_sample(dataset_train[160])
# display_verbose_sample(dataset_train.get_verbose(160))

# %%
dataloaders = {
    "train": DataLoader(dataset=dataset_train, batch_size=16, shuffle=True),
    "val": DataLoader(dataset=dataset_val, batch_size=16, shuffle=True)
}
# %%
data_iter = iter(dataloaders["train"])
# %%
for inputs, labels in data_iter:

    display_traindata(inputs, labels)
    break
# %%
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

training_loop(model, criterion, optimizer_ft, 30, dataloaders, device)

# %%
