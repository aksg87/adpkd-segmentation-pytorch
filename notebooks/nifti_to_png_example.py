# %%
from pathlib import Path
import PIL

import matplotlib.pyplot as plt
import numpy as np
import os

import SimpleITK as sitk


# enable lib loading even if not installed as a pip package or in PYTHONPATH
# also convenient for relative paths in example config files
os.chdir(Path(__file__).resolve().parent.parent)
from adpkd_segmentation.utils.nifti_utils import ( # noqa
    load_nifti,
    nifti_to_png_array,
    process_nifti_dirs
)
from adpkd_segmentation.data.data_utils import path_2dcm_int16  # noqa


# %%
TEST_FOLDER = Path("nifti_tests/annotation4_completed")
study_dirs = [TEST_FOLDER / "4", TEST_FOLDER / "12"]

example_dir = TEST_FOLDER / "4" / WC-ADPKD-____-
dcm_example = example_dir / "DICOM_anon"
nifti_example = dcm_example / "Untitled.nii.gz"


# %%
reader = sitk.ImageSeriesReader()
series_IDs = reader.GetGDCMSeriesIDs(str(dcm_example.resolve()))

# %%
dicom_files = reader.GetGDCMSeriesFileNames(
    str(dcm_example.resolve()), series_IDs[0]
)

# %%
# check one example
nifti_array = load_nifti(nifti_example)[:, :, 35]
png_array = nifti_to_png_array(nifti_array)
dcm_data = path_2dcm_int16(dicom_files[35])

# %%
plt.imshow(nifti_array)
# %%
plt.imshow(png_array)
# %%
plt.imshow(dcm_data)


# %%
# previous data
png = "data_copy/training_data-61-110MR_AX_SSFSE_ABD_PEL_50/WC-ADPKD_AB9-001467-MR1/Ground/00022_2.16.840.1.113669.632.21.1761676154.1761676154.36448706562526961.png"  # noqa
dcm = "data_copy/training_data-61-110MR_AX_SSFSE_ABD_PEL_50/WC-ADPKD_AB9-001467-MR1/DICOM_anon/00022_2.16.840.1.113669.632.21.1761676154.1761676154.36448706562526961.dcm"  # noqa
im = PIL.Image.open(png)
im_array = np.asarray(im)
dcm_array = path_2dcm_int16(dcm)
# %%
plt.imshow(im_array)
# %%
plt.imshow(dcm_array)


# %%
# different data
# more series_IDs
diff_example_dir = Path(
    "nifti_tests/WC-ADPKD_KJ9-002316MR3-AXL FIESTA "
)  # noqa
diff_dcm_example = example_dir / "DICOM_anon"
diff_nifti_example = dcm_example / "Untitled.nii.gz"


# %%
# folder processing test
test_target_dir = Path("nifti_tests/parsed_studies")
process_nifti_dirs(TEST_FOLDER, test_target_dir)

# %%
# checks
test_dcm = "nifti_tests/parsed_studies/WC-ADPKD_AM9-002358MR1-AXL FIESTA /DICOM_anon/00010_2.16.840.1.113669.632.21.136842060.136842060.7615641522567231422.dcm"  # noqa
test_png = test_dcm.replace(".dcm", ".png").replace("DICOM_anon", "Ground")
im = PIL.Image.open(test_png)
im_array = np.asarray(im)
dcm_array = path_2dcm_int16(test_dcm)
# %%
plt.imshow(im_array)
# %%
plt.imshow(dcm_array)


# %%
# REAL RUN
real_source = Path("data/annotation_completed")
real_target = Path("data/processed_studies_sep_26")
process_nifti_dirs(real_source, real_target)

# %%
