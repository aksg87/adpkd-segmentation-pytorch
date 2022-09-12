"""Utilities for converting nifti annotations to png"""

from functools import lru_cache
import os

from cv2 import imwrite
import nibabel as nib
import numpy as np
from pathlib import Path
import SimpleITK as sitk
import shutil

from adpkd_segmentation.datasets.masks import (
    BACKGROUND_INT,
    L_KIDNEY_INT,
    R_KIDNEY_INT,
)

NIFTI_FILE = "Untitled.nii.gz"
DICOM_FOLDER = "DICOM_anon"
GROUND = "Ground"
NIFTI_FOLDER = "Nifti"

png_int_to_itk = {BACKGROUND_INT: 0, R_KIDNEY_INT: 1, L_KIDNEY_INT: 2}


@lru_cache()
def create_png_int_mat(shape):
    return np.stack(
        [
            np.broadcast_to(BACKGROUND_INT, shape=shape).astype("uint8"),
            np.broadcast_to(L_KIDNEY_INT, shape=shape).astype("uint8"),
            np.broadcast_to(R_KIDNEY_INT, shape=shape).astype("uint8"),
        ]
    )


@lru_cache()
def create_nifti_int_mat(shape):
    return np.stack(
        [
            np.broadcast_to(png_int_to_itk[BACKGROUND_INT], shape=shape),
            np.broadcast_to(png_int_to_itk[L_KIDNEY_INT], shape=shape),
            np.broadcast_to(png_int_to_itk[R_KIDNEY_INT], shape=shape),
        ]
    )


def nifti_to_png_array(nifti_array):
    shape = nifti_array.shape
    nifti_array = nifti_array[::-1]
    nifti_array = np.rot90(nifti_array, k=3)

    if x != y:
        shape = (y,x)

    mask = (nifti_array == create_nifti_int_mat(shape)).astype("uint8")
    png_array = (mask * create_png_int_mat(shape)).sum(axis=0)

    return png_array


def load_nifti(nifti_path):
    nimg = nib.load(nifti_path)
    data = nimg.get_data()
    return data


def traverse_folder(folder):
    out = []
    children = [c for c in folder.iterdir() if c.is_dir()]
    if folder.is_dir():
        out.append(folder)
        for c in children:
            out.extend(traverse_folder(c))
    return out


def process_dcm_folder(dcm_folder, target_dir):
    dcm_folder_str = str(dcm_folder)
    parent_name = dcm_folder.parent.name
    target_study = target_dir / parent_name
    nifti_folder = target_study / NIFTI_FOLDER
    ground_folder = target_study / GROUND
    target_dcm_folder = target_study / DICOM_FOLDER
    nifti_target_file = target_study / NIFTI_FOLDER / NIFTI_FILE

    if target_study.exists():
        print(f"Already exists, skipping {target_study}")
        return

    os.makedirs(target_study)
    os.makedirs(nifti_folder)
    os.makedirs(ground_folder)
    os.makedirs(target_dcm_folder)
    shutil.copy(dcm_folder / NIFTI_FILE, nifti_target_file)

    nifti_array = load_nifti(str(nifti_target_file))
    num_annotated = nifti_array.shape[-1]

    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dcm_folder_str)

    # more series, but only one of them is annotated
    if len(series_IDs) > 1:
        correct = None
        for idx, series in enumerate(series_IDs):
            dcm_files = reader.GetGDCMSeriesFileNames(dcm_folder_str, series)
            if len(dcm_files) == num_annotated:
                if correct is None:
                    correct = idx
                else:
                    shutil.rmtree(target_study)
                    print(f"Can't tell which study annotated in {dcm_folder}")
                    raise Exception

        if correct is None:
            shutil.rmtree(target_study)
            print(f"Dicoms and nifti annotations not a match in {dcm_folder}")
            raise Exception
    else:
        correct = 0

    dcm_files = reader.GetGDCMSeriesFileNames(
        dcm_folder_str, series_IDs[correct]
    )
    for idx, dcm in enumerate(dcm_files):
        shutil.copy(dcm, target_dcm_folder)
        png_array = nifti_to_png_array(nifti_array[:, :, idx])
        png_name = Path(dcm.replace(".dcm", ".png")).name
        png_path = str(ground_folder / png_name)
        imwrite(png_path, png_array)


def process_nifti_dirs(source_dir, target_dir):
    if isinstance(source_dir, str):
        source_dir = Path(source_dir)
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    subfolders = traverse_folder(source_dir)
    for folder in subfolders:
        if folder.name == DICOM_FOLDER:
            print(f"Processing study {folder.parent.name}")
            try:
                process_dcm_folder(folder, target_dir)
            except Exception:
                print(f"Couldn't process {folder.parent}")
                continue
