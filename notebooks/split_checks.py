# %%
import os

# `get_labeled` needs a different working dir
# imports not working without PYTHONPATH or package setup
# TODO: refactor later
os.chdir("..")
from data.data_utils import get_labeled, make_dcmdicts
from new_datasets.splits import GenSplit

# %%
dcm_paths = sorted(get_labeled())
dcm2attribs, patient2dcm = make_dcmdicts(tuple(dcm_paths))
all_patient_IDS = list(patient2dcm.keys())

# %%
seed = 1
splitter = GenSplit(seed=seed)
split = splitter(all_patient_IDS)
# %%
print(split["val"])

# %%
for patient in split["val"]:
    sequences = set()
    mrs = set()
    for dcm in patient2dcm[patient]:
        sequences.add(dcm2attribs[dcm]["seq"])
        mrs.add(dcm2attribs[dcm]["MR"])
    print(sequences)
    print(mrs)


# %%
for patient in all_patient_IDS:
    sequences = set()
    mrs = set()
    for dcm in patient2dcm[patient]:
        sequences.add(dcm2attribs[dcm]["seq"])
        mrs.add(dcm2attribs[dcm]["MR"])
    print(sequences)
    print(mrs)

# %%
