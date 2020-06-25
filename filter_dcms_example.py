# TODO: replace with dataset in `new_datasets`
# filtering not fixed in SegmentationDataset

# %%
from sklearn.model_selection import train_test_split

from data_set.datasets import SegmentationDataset
from data.data_utils import (
    make_dcmdicts,
    get_labeled,
    filter_dcm2attribs,
)
from data.link_data import makelinks

from data_set.transforms import T_x, T_y
from model.model_utils import get_preprocessing, preprocessing_fn

# %%

IMG_IDX = 180
# %%
makelinks()
# %%
# find number of patients
dcm2attribs, patient2dcm = make_dcmdicts(tuple(get_labeled()))
all_IDS = range(len(patient2dcm))

# %%

filters = {"seq": "AX SSFSE ABD/PEL"}
dcm2attribs = filter_dcm2attribs(filters, dcm2attribs)


# %%
# train, val, test split--> 85 / 15 / 15.
# Note: Applied to patients not dcm images.
train_IDS, test_IDS = train_test_split(all_IDS, test_size=0.15, random_state=1)
train_IDS, val_IDS = train_test_split(
    train_IDS, test_size=0.176, random_state=1
)

# %%
dataset_train = SegmentationDataset(
    patient_IDS=train_IDS,
    transform_x=T_x,
    transform_y=T_y,
    preprocessing=get_preprocessing(preprocessing_fn),
    filters=filters,
)
dataset_val = SegmentationDataset(
    patient_IDS=val_IDS,
    transform_x=T_x,
    transform_y=T_y,
    preprocessing=get_preprocessing(preprocessing_fn),
    filters=filters,
)

# %%
# View first three examples with attributes
res = list(dataset_train.dcm2attribs.items())[IMG_IDX]
print("examples of attributes for dcm: {}".format(res))
# %%
