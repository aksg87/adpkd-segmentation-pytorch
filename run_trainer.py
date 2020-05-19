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

