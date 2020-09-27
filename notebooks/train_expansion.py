# %%
import json
import os
import random

# enable lib loading even if not installed as a pip package or in PYTHONPATH
# also convenient for relative paths in example config files
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

from adpkd_segmentation.data.data_utils import ( # noqa
    get_labeled,
    make_dcmdicts,
)
from adpkd_segmentation.data.link_data import makelinks  # noqa

# %%
makelinks()
# %%
dcm_paths = sorted(get_labeled())
dcm2attribs, patient2dcm = make_dcmdicts(tuple(dcm_paths))
all_patient_IDS = list(patient2dcm.keys())


# %%
with open("./stratification/strat_split_2020_09_24.json", "r") as f:
    full_split = json.load(f)

# %%
new_train = [
    patient_id
    for patient_id in all_patient_IDS
    if patient_id not in full_split["val"]
    and patient_id not in full_split["test"]
]

# %%
random.seed(42)
random.shuffle(new_train)

# %%
new = {}
new["val"] = full_split["val"]
new["test"] = full_split["test"]
new["train"] = new_train
# %%
with open("./stratification/strat_split_2020_09_24_extended.json", "w") as f:
    json.dump(new, f, indent=4)

# %%
