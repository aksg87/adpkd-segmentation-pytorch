# %%
from collections import defaultdict
import json
import os

import numpy as np
import pandas as pd

from matplotlib.pyplot import hist
from skmultilearn.model_selection import iterative_train_test_split

# enable lib loading even if not installed as a pip package or in PYTHONPATH
# also convenient for relative paths in example config files
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

# %%
from adpkd_segmentation.data.data_utils import (  # noqa
    display_sample,
    get_labeled,
    get_y_Path,
    make_dcmdicts,
    path_2dcm_int16,
    path_2label,
)
from adpkd_segmentation.data.data_utils import (  # noqa
    PATIENT,
    SEQUENCE,
    KIDNEY_PIXELS,
    MR,
    VOXEL_VOLUME,
)
from adpkd_segmentation.data.link_data import makelinks  # noqa

STUDY_TKV = "study_tkv"

# %% needed once for new data
makelinks()

# %%
dcm_paths = sorted(get_labeled())
dcm2attribs, patient2dcm = make_dcmdicts(tuple(dcm_paths))
all_ids = list(patient2dcm.keys())


# TKV checks
# %%
def TKV_update(dcm2attribs):
    studies = defaultdict(int)
    for dcm, attribs in dcm2attribs.items():
        study_id = (attribs[PATIENT], attribs[MR])
        studies[study_id] += attribs[KIDNEY_PIXELS] * attribs[VOXEL_VOLUME]

    for dcm, attribs in dcm2attribs.items():
        tkv = studies[(attribs[PATIENT], attribs[MR])]
        attribs[STUDY_TKV] = tkv

    return studies, dcm2attribs


# %%
studies, dcm2attribs = TKV_update(dcm2attribs)
hist(studies.values(), bins=40)

# %%
hist(np.log(list(studies.values())), bins=40)


# %%
# Patient info
patient_info = set()
for dcm_path, attribs in dcm2attribs.items():
    patient = attribs[PATIENT]
    seq = attribs[SEQUENCE]
    tkv = attribs[STUDY_TKV]
    mr = attribs[MR]
    patient_info.add((patient, seq, mr, tkv))

print(patient_info)


# %%
df = pd.DataFrame.from_records(
    list(patient_info),
    columns=[PATIENT, SEQUENCE, MR, STUDY_TKV],
    index=PATIENT,
).sort_index()

# %%
df.to_csv("./notebooks/patients_2020_11_02.csv")

# %%
print(df.index.value_counts())

# %%
print(df.seq.value_counts())

# %%
print(df.study_tkv.describe())
print(np.log(df.study_tkv).describe())


# %%
def create_label_arrays(patient_info, all_ids):
    patient_to_label = {}
    for id_ in all_ids:
        patient_to_label[id_] = np.zeros(8, dtype=np.uint8)
    for patient, seq, mr, tkv in patient_info:
        # sequence labeling
        # the same patient can have more
        if "AX SSFSE HR KIDNEYS" == seq:
            patient_to_label[patient][0] = 1
        elif "AXL FIESTA" == seq:
            patient_to_label[patient][1] = 1
        elif "PEL" in seq:
            patient_to_label[patient][2] = 1
        else:
            patient_to_label[patient][3] = 1
        # LOG TKV category
        # 13.69 to 14.74 interquartile range
        # 14.19 median
        log_tkv = np.log(tkv)
        if log_tkv < 13.7:
            patient_to_label[patient][4] = 1
        elif 13.7 <= log_tkv < 14.2:
            patient_to_label[patient][5] = 1
        elif 14.2 <= log_tkv < 14.7:
            patient_to_label[patient][6] = 1
        elif 14.7 <= log_tkv:
            patient_to_label[patient][7] = 1

    return patient_to_label


# %%
patient2label = create_label_arrays(patient_info, all_ids)

# %%
labels = [patient2label[id_] for id_ in all_ids]
X = np.array(all_ids)[..., np.newaxis]
y = np.stack(labels)

print(y.shape)
print(len(all_ids))
print(X.shape)

# %%
# Split to train, val, test
TRAIN = 0.7
VAL = 0.15
TEST = 0.15
np.random.seed = 42

X_train_val, y_train_val, X_test, y_test = iterative_train_test_split(
    X, y, test_size=TEST
)

X_train, y_train, X_val, y_val = iterative_train_test_split(
    X_train_val, y_train_val, test_size=VAL / (TRAIN + VAL)
)

# %%
print("Sizes: ", len(X_train), len(X_val), len(X_test))

# %%
print(y_train)
print(y_val)
print(y_test)

# %%
print(df[df.index.isin(X_test.squeeze())])
print(df[df.index.isin(X_val.squeeze())])

# %%
split_dict = {
    "train": list(X_train.squeeze()),
    "val": list(X_val.squeeze()),
    "test": list(X_test.squeeze()),
}


# %%
# move patient with transplanted kidneys to train
# out of scope
# outlier = WC-ADPKD-____-
# split_dict["val"].remove(outlier)
# split_dict["train"].append(outlier)

# %%
with open("./stratification/strat_split_2020_11_03.json", "w") as f:
    json.dump(split_dict, f, indent=4)

# %%
