import nibabel as nib
import SimpleITK as sitk
import numpy as np
import cv2
import os
import shutil

"""
To run this script, define the source_path where ITK annotations are exported
folder structure is as follows:

Patient_directory
--DICOM_anon_directory
--Untitled.nii.gz

After running the script:
- new directory will be greated called Ground with png masks
- nifti file will be moved to a Nifti directory

"""
source_path = "/PKD-DATA/training_data_AX_SSFSE_ABD_PEL_30"

itkpixel2gray = {
        "0": 0,
        "1": 128,
        "2": 189,
        "3": 63,
        "4": 252,
    }


def load_nifti(nifti_path):
    nimg = nib.load(nifti_path)
    data = nimg.get_data()
    dimension = data.shape

    return data, dimension

def create_gray_array(np_data, dim1, dim2):

    pixels = np_data.tolist()
    png_pixel_data = []
    for pixel in pixels:
        for n, i in enumerate(pixel):
            pixel[n] = itkpixel2gray.get(str(i))
        png_pixel_data.append(pixel)
    np_data = np.array(png_pixel_data)
    np_data = np.reshape(png_pixel_data, (dim1, dim2))
    return np_data.astype(np.uint8)

def nifti_to_Png(dicom_files,nifti_path):

    data, _ = load_nifti(nifti_path)
    if(data is not None and data.ndim == 3):
        for index,dcm in enumerate(dicom_files):
            dicom_annon_path = os.path.dirname(dcm)
            ground_path = dicom_annon_path.replace("DICOM_anon","Ground")
            dcm_file_name = os.path.basename(dcm).replace(".dcm",".png")
            os.makedirs(ground_path,exist_ok=True)
            nifti_data = data[:, :, index]
            nifti_data = nifti_data[::-1]

            # rotate image orientation due to nifti format
            rgb_data = np.rot90(nifti_data, k=3)
            gray_data = create_gray_array(
                        rgb_data, nifti_data.shape[0], nifti_data.shape[1])

            png_path = "{}/{}".format(str(ground_path),dcm_file_name)
            cv2.imwrite(png_path, gray_data)

"""lines the dicom files with nifti images for png conversion"""
def format2name_export2png(dicom_path,nifti_path):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_path)

    for series_id in series_IDs:
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_path, series_id)

        nifti_to_Png(dicom_files,nifti_path)

def read_folders(source_path):

    with os.scandir(source_path) as roots:
        for root in roots:
            if root.is_dir():
                    folder_path = root.path
                    nifti_pre_move = str(folder_path)+"/Untitled.nii.gz"
                    nifti_post_move = str(folder_path)+"/Nifti/Untitled.nii.gz"
                    dicom_path = str(folder_path)+"/DICOM_anon"

                    if(os.path.isfile(nifti_pre_move)): #nifti to dedicated folder
                        os.makedirs(str(folder_path)+"/Nifti",exist_ok=True)
                        shutil.move(nifti_pre_move, nifti_post_move)
                        format2name_export2png(dicom_path, nifti_post_move)

                    elif(os.path.isfile(nifti_post_move)):
                        format2name_export2png(dicom_path, nifti_post_move)

                    else:
                        continue
            break # uncomment for single file test
read_folders(source_path)
