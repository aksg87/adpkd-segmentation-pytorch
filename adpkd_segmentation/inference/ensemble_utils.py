from typing import Union, List
import os
from pathlib import Path
import nibabel as nib
import numpy as np


string_path = Union[str, Path]  # Can be a string or pathlib Path


def get_tail(dir: string_path, depth: int) -> str:
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


def get_scan(intermediate_folder: str, dir: string_path) -> str:
    target_depth = 1
    if get_tail(dir, target_depth) == intermediate_folder:
        target_depth += 1
    return get_tail(dir, target_depth)


def grab_organ_dirs(
    organ_paths: List[string_path],
    ensemble_mode: str,
    organ_name: List[str],
    pred_filename: str,
) -> dict:
    """Given the initial parent path,
    we will provide the directories to
    all files + scans"""
    path_dict = {}
    if ensemble_mode == "ensemble addition":
        for organ, organ_path in zip(organ_name, organ_paths):
            path_dict[organ] = list(
                Path(organ_path).glob(f"**/*{pred_filename}")
            )
        return path_dict


def mask_add(
    scan_iter: int,
    mask_directory_dict: dict,
    organ_name: List[str],
    add_organ_color: List[int],
) -> np.ndarray:
    for idO, organ, add_color in zip(
        range(len(organ_name)),
        organ_name,
        add_organ_color,
    ):
        load_dir = mask_directory_dict[organ][scan_iter]
        tmp_nii = nib.load(load_dir)
        tmp_arr = np.array(tmp_nii.dataobj)
        if idO == 0:
            temp_combine = np.zeros(np.shape(tmp_arr))
        temp_combine += add_color * tmp_arr

    return np.uint16(temp_combine)


def addition_ensemble(
    scan_iter: int,
    mask_directory_dict: dict,
    organ_name: List[str],
    add_organ_color: List[str],
) -> np.ndarray:
    """Given a dictionary that temporarily holds
    the organ paths and scan index,"""
    overlap_mask = mask_add(
        scan_iter=scan_iter,
        mask_directory_dict=mask_directory_dict,
        organ_name=organ_name,
        add_organ_color=add_organ_color,
    )
    return overlap_mask
