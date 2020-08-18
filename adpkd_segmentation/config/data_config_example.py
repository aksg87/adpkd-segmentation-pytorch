from pathlib import Path

# %%
# modify according to the location of your data

# after modification, rename and place this file as
# adpkd_segmentation.data.data_config.py
dataroot = Path("/root/data/")

labeled_dirs = [
    dataroot / "training-data-01-60MR",
    dataroot / "training_data-61-110MR_AX_SSFSE_ABD_PEL_50",
]

unlabeled_dirs = [dataroot / "unlabelled_data"]
