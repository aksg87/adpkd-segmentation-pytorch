# %% 
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from data_set.datasets import SegmentationDataset
from data.data_utils import make_dcmdicts, get_labeled, masks_to_colorimg, display_sample
from data.link_data import makelinks

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
dataset_train = SegmentationDataset(patient_IDS=train_IDS)
dataset_val = SegmentationDataset(patient_IDS=val_IDS)
