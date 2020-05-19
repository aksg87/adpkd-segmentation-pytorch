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
from data.data_utils import *
from data.link_data import makelinks

from data_set.datasets import *
from data_set.transforms import T_x, T_y 
from trainer.trainer import *

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
dataset_train = SegmentationDataset(patient_IDS=train_IDS, transform_x=T_x, transform_y=T_y)
dataset_val = SegmentationDataset(patient_IDS=val_IDS, transform_x=T_x, transform_y=T_y)

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
