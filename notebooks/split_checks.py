# %%
import os

# enable lib loading even if not installed as a pip package or in PYTHONPATH
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

from adpkd_segmentation.data.data_utils import get_labeled, make_dcmdicts  # noqa
from adpkd_segmentation.datasets.splits import GenSplit  # noqa

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
print(split["test"])

# %%
for patient in split["test"]:
    sequences = set()
    mrs = set()
    for dcm in patient2dcm[patient]:
        sequences.add(dcm2attribs[dcm]["seq"])
        mrs.add(dcm2attribs[dcm]["MR"])
    print(sequences)
    print(mrs)


# %%
print(split["train"])

# %%
for patient in split["train"]:
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
