
# %%
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import nibabel as nib
import pydicom as dcm
from collections import defaultdict

import os
os.chdir(Path(__file__).resolve().parent.parent)
from adpkd_segmentation.data.data_config import dataroot  # noqa
# %%


def nii2dcm_sequences(nii_file):
    sequence_dict = defaultdict(list)

    dcm_files = list(nii_file.parent.glob("*.dcm"))
    dcm_files = [dcm.read_file(d) for d in dcm_files]
    dcm_files.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    for d in dcm_files:
        sequence_dict[d.SeriesNumber].append(d)

    if len(sequence_dict) > 1:
        seq_lens = [len(v) for v in sequence_dict.values()]
        if len(set(seq_lens)) != len(seq_lens):
            return [v[0].SeriesNumber for v in sequence_dict.values()]

    return []


def nii2dcm(nii_file, sequence_num=None):
    dcm_files = list(nii_file.parent.glob("*.dcm"))
    dcm_files = [dcm.read_file(d) for d in dcm_files]
    dcm_files.sort(key=lambda x: int(x.ImagePositionPatient[2]))

    if sequence_num is not None:
        dcm_files = [
            d for d in dcm_files if d.SeriesNumber == sequence_num
        ]
    dcm_arrs = [d.pixel_array for d in dcm_files]

    label_arr = nib.load(nii_file).get_fdata().T

    return np.stack(dcm_arrs), label_arr

# %%


def plot_image_bar(image, label=None, title=None, standardize=True):

    ax = plt.subplot(111)
    im = ax.imshow(
        image, cmap="gray", origin='lower', alpha=1
    )

    if label is not None:
        if standardize is True:
            label[label > 0] = 1
        im = ax.imshow(
            label, cmap="viridis", origin='lower', alpha=0.5
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    if title is not None:
        ax.set_title(title, size=8)

    plt.show()
# %%


def plot_nii_dcm(nii_file, axis=0, seq_list=None):

    if seq_list is not None:
        for seq in seq_list:
            dcms, label = nii2dcm(nii_file, sequence_num=seq)
            idx = 3*len(dcms) // 4
            plot_image_bar(
                dcms[idx],
                label[idx],
                title=nii_file.parent.parent.name + " " + str(seq)
            )
    else:
        dcms, label = nii2dcm(nii_file)
        plot_image_bar(
            np.sum(dcms, axis=axis),
            label=np.sum(label, axis=axis),
            title=nii_file.parent.parent.name
        )


# %%
completed = dataroot / "annotation_completed"
niis = list(completed.glob("**/*.gz"))

# %%

# get list of duplicates with same series len
checks = []
for i, n in enumerate(niis):
    res = nii2dcm_sequences(n)
    if res:
        checks.append((n, res))

# %%

# dispay series to identify which matches mask

for idx, c in enumerate(checks):
    print(idx, c[0])
    plot_nii_dcm(c[0], seq_list=c[1])


# %%

# manually delete a series by number
def remove_dicom_series(nii_file, SeriesNumber):
    dcm_files = list(nii_file.parent.glob("*.dcm"))
    # dcm_files = [dcm.read_file(d) for d in dcm_files]

    print(f"there are {len(dcm_files)}...")
    
    dcm_files = [
        d for d in dcm_files if dcm.read_file(d).SeriesNumber == SeriesNumber
    ]

    print(f"removing {len(dcm_files)}...")

    for d in dcm_files:
        os.remove(d)

# %%
