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


def reassign_color(
    input_array: np.ndarray,
    old_colors: List[int],
    new_colors: List[int],
) -> np.ndarray:
    temp_arr = input_array
    for old_color, new_color in zip(old_colors, new_colors):
        temp_arr[input_array == old_color] = new_color
    # end
    return np.uint16(temp_arr)


def remap_kidney(
    affine_path: string_path,
    input_array: np.ndarray,
    kidney_side: str,
    addition_color: int,
    viewer_color: int,
) -> np.ndarray:
    ref_img = nib.load(affine_path)
    aff = ref_img.affine  # TODO: Look this over before testing
    F = aff[0:3, 0:3]
    S = aff[0:3, 3]
    I, J, K = np.mgrid[
        0 : input_array.shape[0],
        0 : input_array.shape[1],
        0 : input_array.shape[2],
    ]
    F_xi = F[0, 0]
    F_xj = F[0, 1]
    F_xk = F[0, 2]
    S_0x = S[0]
    x = F_xi * I + F_xj * J + F_xk * K + S_0x  # R-L component
    x_unique = np.unique(x)
    xmin = min(x_unique)
    xmax = max(x_unique)
    xmag = xmax - xmin
    xhalf = xmin + 0.5 * xmag
    tmp_img = input_array
    tmp_half = np.zeros(input_array.shape)
    if kidney_side == "right":
        tmp_half[x > xhalf] = 1
    elif kidney_side == "left":
        tmp_half[x < xhalf] = 1
    # end
    tmp_kidney = np.zeros(input_array.shape)
    tmp_kidney[input_array == addition_color] = 1
    tmp_sum = tmp_half + tmp_kidney
    tmp_img[tmp_sum == 2] = viewer_color
    return np.uint16(tmp_img)  # TODO: Testing when I can


def addition_ensemble(
    scan_iter: int,
    mask_directory_dict: dict,
    organ_name: List[str],
    add_organ_color: List[int],
    overlap_colors: List[int],
    adjudicated_colors: List[int],
    old_organ_colors: List[int],
    new_organ_colors: List[int],
    selected_kidney_side: str,
    kidney_addition_color: int,
    kidney_viewer_color: int,
) -> np.ndarray:
    """Given a dictionary that temporarily holds
    the organ paths and scan index,"""
    overlap_mask = mask_add(
        scan_iter=scan_iter,
        mask_directory_dict=mask_directory_dict,
        organ_name=organ_name,
        add_organ_color=add_organ_color,
    )
    remap_organs = reassign_color(
        overlap_mask,
        old_colors=overlap_colors,
        new_colors=adjudicated_colors,
    )
    remap_kidneys = reassign_color(
        remap_organs, old_colors=old_organ_colors, new_colors=new_organ_colors
    )
    output_mask = remap_kidney(
        affine_path=mask_directory_dict[organ_name[0]][scan_iter],
        input_array=remap_kidneys,
        kidney_side=selected_kidney_side,
        addition_color=kidney_addition_color,
        viewer_color=kidney_viewer_color,
    )
    return output_mask
