import os
from pathlib import Path
import nibabel as nib
import numpy as np


def get_tail(dir=".", depth=1):
    """
    Given a folder path OR directory, return
    the folder given the depth.
    Note: depth = 1 denotes the newest folder
    in the tree.
    """
    t_head = dir
    num_depth = depth
    if os.path.isfile(dir):
        num_depth += 1
    #
    idx = 0
    while idx < num_depth:  # test
        t_head, t_tail = os.path.split(t_head)
        idx += 1
    #
    return t_tail  # Tested, good


def get_scan(config, dir="."):
    """This is designed for the output folder pattern in
    ADPKD-segmentation-pytorch (Goel et al. Radiology AI). At inference, the
    nifti files are saved in
    .../saved_inference/repo_name/patient_ID/scan_folder/ITKSNAP_DCM_NIFTI/pred.vol.nii
    You can find Dr. Goel's repo at:
    https://github.com/aksg87/adpkd-segmentation-pytorch.git
    And you can find his deployment:
    web: https://pubs.rsna.org/doi/10.1148/ryai.210205
    doi: https://doi.org/10.1148/ryai.210205
    """
    target_depth = 1
    if get_tail(dir, target_depth) == config["youngest_child"]:
        target_depth += 1
    return get_tail(dir, target_depth)


def grab_organ_dirs(iter_dict, config):
    """Given the initial parent path,
    we will provide the directories to
    all files + scans"""
    path_dict = {}
    if config["mode"] == "ensemble addition":
        for organ, organ_path in zip(
            config["organ_name"], iter_dict["organ_paths"]
        ):
            path_dict[organ] = list(
                Path(organ_path).glob(f"**/*{config['pred_vol']}")
            )
        return path_dict  # TEST THIS
    # To Do: if config['mode'] = "ensemble softmax"


def mask_add(iter_dict, config):
    idScan = iter_dict["idS"]  # Scan index
    for idO, organ, add_color in zip(
        range(len(config["organ_name"])),
        config["organ_name"],
        config["add_organ_color"],
    ):
        load_dir = iter_dict["mask_directories"][organ][idScan]
        tmp_nii = nib.load(load_dir)
        tmp_arr = np.array(tmp_nii.dataobj)
        if idO == 0:
            temp_combine = np.zeros(np.shape(tmp_arr))
        temp_combine += add_color * tmp_arr

    return np.uint16(temp_combine)


def addition_ensemble(iter_dict, config):
    """Given a dictionary that temporarily holds
    the organ paths and scan index,"""
    # Current commit: finalize mask_add
    # Next commit:
    #   -overlap_repaint
    #   -organ_repaint
    iter_dict["mask_directories"] = grab_organ_dirs(iter_dict, config)
    overlap_mask = mask_add(iter_dict, config)
    return overlap_mask  # For testing purposes -- v1_002 only
    # Coming soon -- v1_003
    # process_org_color_mask = overlap_repaint(overlap_mask,config,other_args...)
    # comb_mask = organ_repaint(process_org_color_mask,config)
    # return comb_mask
