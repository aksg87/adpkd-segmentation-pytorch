from typing import Union, List, Dict
import os
from pathlib import Path
import nibabel as nib
import numpy as np
import json
import shutil
import torch
import pydicom
import pandas as pd
from inference_utils import IOP_IPP_dicomsort
import SimpleITK as sitk
import cv2
import albumentations
from tqdm import tqdm

IOP = "IOP"
IPP = "IPP"
IPP_dist = "IPP_dist"
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
    aff = ref_img.affine  # From SimpleITK nifti generation
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
    return np.uint16(tmp_img)


def scan_list(dicom_list: List[string_path], rules_dict: Dict[str, str]):
    input_folders = []
    scans = []
    representative_dicoms = []
    for dicom in dicom_list:
        dicom_info = pydicom.dcmread(dicom)
        dicom_scan = dicom_info.SeriesDescription
        for key, value in rules_dict.items():
            dicom_scan = dicom_scan.replace(key, value)

        tmp_folder, _ = os.path.split(dicom)

        if dicom_scan not in scans:
            input_folders.append(tmp_folder)
            scans.append(dicom_scan)
            representative_dicoms.append(dicom)

    return input_folders, scans, representative_dicoms


def select_sequence_key(input_dicom: string_path) -> str:
    key_list = ["T2", "SSFP", "T1"]
    dicom_info = pydicom.dcmread(input_dicom)
    dicom_acquisition = dicom_info.MRAcquisitionType
    dicom_sequence_type = dicom_info.ScanningSequence
    if dicom_acquisition == "3D":
        return key_list[2]
    elif dicom_acquisition == "2D":
        if dicom_sequence_type == "SE":
            return key_list[0]
        elif dicom_sequence_type == "GR":
            return key_list[1]
        elif dicom_sequence_type == "IR":
            return key_list[0]
        elif dicom_sequence_type == "RM":
            return key_list[0]
        else:
            return key_list[0]


def select_plane_key(
    input_dicom: string_path, reference_directions: str, plane_keys: List[str]
) -> str:
    """
    For a given input of dicoms, select the key that corresponds to the
    scan plane from the DICOM header. Use the inner product to find the plane
    """
    plane_vecs = np.matrix(reference_directions)
    dicom_info = pydicom.dcmread(input_dicom)
    dicom_orientation = dicom_info.ImageOrientationPatient
    ori_vec = np.array(dicom_orientation)
    patient_direction = np.cross(
        ori_vec[0:3], ori_vec[3:6]
    )  # Cartesian direction of patient plane
    dot_ori = np.zeros(np.shape(plane_vecs)[0])
    for ind in range(np.shape(plane_vecs)[0]):
        ref_basis = np.cross(
            plane_vecs[ind, 0:3], plane_vecs[ind, 3:6]
        ).flatten()  # Reference direction (cartesian basis)
        # |<u,v>|/|v| -- measure of most axis alignment
        dot_ori[ind] = np.absolute(np.dot(patient_direction, ref_basis))

    max_ind = np.argmax(dot_ori)
    max_val = dot_ori[max_ind]
    print(f"Maximum dot product of orientations: {max_val}")
    return plane_keys[max_ind]


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


def binary_inference_to_disk(
    dataloader,
    model,
    device,
    binarize_func,
    save_dir="./saved_inference",
):
    """
    Generates inferences from InferenceDataloader.

    Args:
        dataloader (dataloader): Dataloader instance for an InferenceDataset.
        model (model): Dataloader instance.
        device (device): Device instance.
        binarize_func (function): Binarizing function.
        save_dir (str, optional): Directory to save inference. Defaults to "./saved_inference".
        model_name (str, optional): Name of model. Defaults to "model".

    """
    dataset = dataloader.dataset
    output_idx_check = (
        hasattr(dataloader.dataset, "output_idx")
        and dataloader.dataset.output_idx
    )

    assert (
        output_idx_check is True
    ), "output indexes are required for the dataset"

    for batch_idx, output in enumerate(dataloader):

        x_batch, idxs_batch = output
        x_batch = x_batch.to(device)

        with torch.no_grad():

            # get verbose returns (sample, dcm_path, attributes dict)
            dcm_file_paths = [
                Path(dataset.get_verbose(idx)[1]) for idx in idxs_batch
            ]

            dcm_file_names = [
                Path(dataset.get_verbose(idx)[1]).stem for idx in idxs_batch
            ]

            file_attribs = [dataset.get_verbose(idx)[2] for idx in idxs_batch]

            # Inference
            y_batch_hat = model(x_batch)
            y_batch_hat_binary = binarize_func(y_batch_hat)

            for dcm_path, dcm_name, file_attrib, img, logit, pred in zip(
                dcm_file_paths,
                dcm_file_names,
                file_attribs,
                x_batch,
                y_batch_hat,
                y_batch_hat_binary,
            ):
                out_dir = Path(save_dir) / dcm_name
                out_dir.parent.mkdir(parents=True, exist_ok=True)

                # Save the output
                np.save(str(out_dir) + "_img", img.cpu().numpy())
                np.save(str(out_dir) + "_logit", logit.cpu().numpy())
                np.save(str(out_dir) + "_pred", pred.cpu().numpy())
                shutil.copy(
                    dcm_path, out_dir.parent / (out_dir.name + "_DICOM.dcm")
                )

                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        else:
                            return super(NpEncoder, self).default(obj)

                # get resize transform within compose object
                Resize = albumentations.augmentations.geometric.resize.Resize
                # Resize = albumentations.augmentations.transforms.Resize
                transform_resize = next(
                    v
                    for v in dataloader.dataset.augmentation.transforms
                    if isinstance(v, Resize)
                )
                assert (
                    transform_resize is not None
                ), "transform_resize must be defined"

                file_attrib["transform_resize_dim"] = (
                    transform_resize.height,
                    transform_resize.width,
                )

                attrib_json = json.dumps(file_attrib, cls=NpEncoder)
                f = open(str(out_dir) + "_attrib.json", "w")
                f.write(attrib_json)
                f.close()


def argmax_ensemble(
    scan_list: List[string_path],
    output_folder: string_path,
    organ_name: List[str],
    index_classes: List[int],
    itk_colors: List[int],
):
    # I need: class indeces, output integers
    numscans = len(scan_list)
    for idScan, scan in enumerate(scan_list):
        _, individual_scan = os.path.split(scan)
        one_hot = []
        print(f"Ensembling {idScan+1}/{numscans}")
        for i, organ in enumerate(organ_name):
            organ_path = os.path.join(scan, organ)
            organ_logits = Path(organ_path).glob("*_logit*")
            organ_logits = sorted(organ_logits, key=lambda x: x.name)
            npy_organ_logits = [
                np.squeeze(np.load(Path(p))) for p in organ_logits
            ]
            npy_logit_vol = np.stack(npy_organ_logits, axis=-1)

            if i == 0:
                one_hot.append(np.zeros(npy_logit_vol.shape))
                output_scan = Path(output_folder) / individual_scan
                output_scan.mkdir(parents=True, exist_ok=True)
                dcm_paths = Path(organ_path).glob("*.dcm")
                for dcm_path in dcm_paths:
                    shutil.copy(dcm_path, output_scan)

            one_hot.append(npy_logit_vol)

        one_hot = np.stack(one_hot, -1)
        one_hot_dim = len(one_hot.shape) - 1
        one_hot = torch.tensor(one_hot)
        softmax_func = torch.nn.Softmax(dim=one_hot_dim)
        prediction_softmax = softmax_func(one_hot)
        prediction_map = torch.argmax(prediction_softmax, dim=one_hot_dim)
        prediction_map = prediction_map.numpy()
        ref_map = prediction_map

        for max_index, itk_color in zip(index_classes, itk_colors):
            prediction_map[ref_map == max_index] = itk_color

        dcms = output_scan.glob("*.dcm")
        dcms = sorted(dcms, key=lambda x: x.name)
        for i, dcm in enumerate(dcms):
            pred_slice = np.uint16(prediction_map[:, :, i])
            file_name = str(dcm).replace("_DICOM.dcm", "")
            file_name = f"{file_name}_multi_pred"
            np.save(file_name, pred_slice)

        all_dcms = Path(scan).glob("**/*.dcm")
        for each_dcm in all_dcms:
            os.remove(each_dcm)
        all_dcms = []
        all_npys = Path(scan).glob("**/*.npy")
        for each_npy in all_npys:
            os.remove(each_npy)
        all_npys = []
        all_jsons = Path(scan).glob("**/*.json")
        for each_json in all_jsons:
            os.remove(each_json)


def ensemble_to_nifti(
    output_scan_list: List[string_path],
    selected_kidney_side: str,
    kidney_ensemble_color: int,
    kidney_side_color: int,
    inverse_crop_ratio=1,
):
    for scan in tqdm(output_scan_list):
        preds = Path(scan).glob("*multi_pred.npy")
        dcm_paths = Path(scan).glob("*.dcm")

        preds = sorted(preds, key=lambda x: x.name)
        dcm_paths = sorted(dcm_paths, key=lambda x: x.name)

        dcms = [pydicom.read_file(p) for p in dcm_paths]

        IOPs = [d.ImageOrientationPatient for d in dcms]
        IPPs = [d.ImagePositionPatient for d in dcms]

        data = {"preds": preds, "dcm_paths": dcm_paths, IOP: IOPs, IPP: IPPs}
        sorted_df = IOP_IPP_dicomsort(pd.DataFrame(data))

        # Use SITK to generate numpy from dicom header
        reader = sitk.ImageSeriesReader()
        sorted_dcm_paths = [str(p) for p in sorted_df["dcm_paths"]]
        reader.SetFileNames(sorted_dcm_paths)
        errors = []

        try:
            image_3d = reader.Execute()
        except Exception as e:
            errors.append(f"error:{str(e)}\n path:{dcm_paths[0]}")

        out_dir = dcm_paths[0].parent
        dcm_save_name = "dicom_vol.nii"
        pred_save_name = "pred_vol.nii"

        sitk.WriteImage(image_3d, str(out_dir / dcm_save_name))

        # Load saved nii volume into nibabel object
        dcm_nii_vol = nib.load(out_dir / dcm_save_name)

        npy_preds = [np.squeeze(np.load(Path(p))) for p in sorted_df["preds"]]

        # reverse center crop -- idx 0 to get shape
        pad_width = (
            (npy_preds[0].shape[0] * inverse_crop_ratio)
            - (npy_preds[0].shape[0])
        ) / 2
        pad_width = round(pad_width)

        npy_reverse_crops = [np.pad(pred, pad_width) for pred in npy_preds]

        # resize predictions to match dicom
        x_y_dim = dcm_nii_vol.get_fdata().shape[0:2]  # shape in x,y,z
        resized_preds = [
            cv2.resize(orig, (x_y_dim), interpolation=cv2.INTER_NEAREST)
            for orig in npy_reverse_crops
        ]

        corrected_transpose = [np.transpose(r) for r in resized_preds]

        # convert 2d npy to 3d npy volume
        npy_pred_vol = np.stack(corrected_transpose, axis=-1).astype(np.uint16)

        # Recolor the selected kidney
        dcm_path = str(out_dir / dcm_save_name)
        npy_pred_vol = remap_kidney(
            dcm_path,
            npy_pred_vol,
            selected_kidney_side,
            kidney_ensemble_color,
            kidney_side_color,
        )

        # create nifti mask for predictions
        dicom_header = dcm_nii_vol.header.copy()
        pred_nii_vol = nib.Nifti1Image(npy_pred_vol, None, header=dicom_header)
        nib.save(pred_nii_vol, out_dir / pred_save_name)

        print(f"Wrote to: {Path(str(out_dir / dcm_save_name))}")
        print("Deleting dicoms and numpy arrays...")
        for pred, dcm_path in zip(preds, dcm_paths):
            os.remove(pred)
            os.remove(dcm_path)
