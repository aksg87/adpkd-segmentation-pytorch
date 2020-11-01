
# %%
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import nibabel as nib
import pydicom as dcm

import os
os.chdir(Path(__file__).resolve().parent.parent)
from adpkd_segmentation.data.data_config import dataroot  # noqa
# %%


def nii2dcm(nii_file):
    dcm_files = list(nii_file.parent.glob("*.dcm"))
    dcm_files = [dcm.read_file(d) for d in dcm_files]
    dcm_files.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    dcm_arrs = [d.pixel_array for d in dcm_files]

    label_arr = nib.load(nii_file).get_fdata().T

    return np.stack(dcm_arrs), label_arr

# %%


def plot_image_bar(image, label=None, title=None):

    ax = plt.subplot(111)
    im = ax.imshow(
        image, cmap="gray", origin='lower', alpha=1
    )

    if label is not None:
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


def plot_nii_dcm(nii_file, axis=0):
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
plot_nii_dcm(niis[0])
plot_nii_dcm(niis[0], axis=1)

# %%
# displays 3d sum of masks over dcms for error analysis
start = 160
for i in range(start, start+20):
    plot_nii_dcm(niis[i])
    plot_nii_dcm(niis[i], axis=1)

# %%
