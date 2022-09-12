# %%

import functools
import glob

from collections import defaultdict, OrderedDict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom

from PIL import Image

from adpkd_segmentation.data.data_config import LABELED, UNLABELED


MIN_VALUE = "global_min_int16"
MAX_VALUE = "global_max_int16"
MIN_IMAGE_VALUE = "min_image_int16"
MAX_IMAGE_VALUE = "max_image_int16"

PATIENT = "patient"
SEQUENCE = "seq"
KIDNEY_PIXELS = "kidney_pixels"
MR = "MR"

VOXEL_VOLUME = "vox_vol"
DIMENSION = "dim"
STUDY_TKV = "study_tkv"


# %%
def int16_to_uint8(int16):
    return cv2.normalize(
        int16,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )


def normalize(int16, attribs):
    min_value = attribs[MIN_VALUE]
    max_value = attribs[MAX_VALUE]
    new_min = 0
    new_max = 255
    # linear scaling
    scaled = (new_max - new_min) / (max_value - min_value) * (
        int16 - min_value
    ) + new_min
    return scaled.astype(dtype=np.uint8)


def add_patient_sequence_min_max(dcm2attribs):
    # modifies dcm2attribs with additional info
    patient_seq_dict_mins = {}
    patient_seq_dict_maximums = {}

    for dcm, attribs in dcm2attribs.items():
        patient = attribs[PATIENT]
        seq = attribs[SEQUENCE]
        im_min_value = attribs[MIN_IMAGE_VALUE]
        im_max_value = attribs[MAX_IMAGE_VALUE]
        current_min = patient_seq_dict_mins.get((patient, seq), float("inf"))
        current_max = patient_seq_dict_maximums.get(
            (patient, seq), float("-inf")
        )
        if im_min_value <= current_min:
            patient_seq_dict_mins[(patient, seq)] = im_min_value
        if im_max_value >= current_max:
            patient_seq_dict_maximums[(patient, seq)] = im_max_value

    # store global min and max for each dcm
    for dcm, attribs in dcm2attribs.items():
        patient = attribs[PATIENT]
        seq = attribs[SEQUENCE]
        attribs[MIN_VALUE] = patient_seq_dict_mins[(patient, seq)]
        attribs[MAX_VALUE] = patient_seq_dict_maximums[(patient, seq)]


def TKV_update(dcm2attribs):
    studies = defaultdict(int)
    for dcm, attribs in dcm2attribs.items():
        study_id = (attribs[PATIENT], attribs[MR])
        studies[study_id] += attribs[KIDNEY_PIXELS] * attribs[VOXEL_VOLUME]

    for dcm, attribs in dcm2attribs.items():
        tkv = studies[(attribs[PATIENT], attribs[MR])]
        attribs[STUDY_TKV] = tkv

    return studies, dcm2attribs


def tensor_dict_to_device(tensor_dict, device):
    out = {}
    for k, v in tensor_dict.items():
        out[k] = v.to(device)
    return out


class NormalizePatientSeq:
    @staticmethod
    def __call__(int16, attribs):
        return normalize(int16, attribs)

    @staticmethod
    def update_dcm2attribs(dcm2attribs):
        print(
            "Adding global min and max image value for each "
            "(patient, sequence) tuple"
        )
        add_patient_sequence_min_max(dcm2attribs)


def get_dcms_paths(dir_list):

    all_files = []

    for study_dir in dir_list:
        print("processing {} ".format(study_dir))

        files = glob.glob("{}/**/*.dcm".format(study_dir), recursive=True)
        all_files.extend(files)

        print("total files... --> {} \n".format(len(all_files)))

    return all_files


def get_labeled():
    dcms = glob.glob("{}/*.dcm".format(LABELED))
    return dcms


def get_unlabeled():
    dcms = glob.glob("{}/*.dcm".format(UNLABELED))
    return dcms


def get_y_Path(x):
    """Get label path from dicom path"""

    if isinstance(x, str):
        x = Path(x)

    y = str(x.absolute()).replace("DICOM_anon", "Ground")
    y = y.replace(".dcm", ".png")
    y = Path(y)

    return y


def path_2dcm_int16(fname):

    if not isinstance(fname, str):
        fname = str(fname)
    dcm = pydicom.dcmread(fname)
    return dcm.pixel_array


def path_2label(fname):

    if not isinstance(fname, str):
        fname = str(fname)
    label = Image.open(fname)
    return np.array(label)


def dcm_attributes(dcm, label_status=True, WCM=True):

    attribs = {}

    # dicom header attribs
    pdcm = pydicom.dcmread(dcm)
    arr_int16 = pdcm.pixel_array

    # TODO refactor for this PatientID clause
    # WCM PatientIDs are of length 10
    if WCM is True and len(pdcm.PatientID) != 10:
        attribs[PATIENT] = pdcm.PatientID[:-3]
        attribs[MR] = pdcm.PatientID[-3:]
        attribs[SEQUENCE] = pdcm.SeriesDescription
    else:
        attribs[PATIENT] = pdcm.PatientID
        attribs[MR] = pdcm.SeriesDescription
        attribs[SEQUENCE] = pdcm.SeriesDescription

    attribs[MIN_IMAGE_VALUE] = arr_int16.min()
    attribs[MAX_IMAGE_VALUE] = arr_int16.max()

    # pixels in mask --> kidney
    if label_status is True:
        label = np.array(Image.open(get_y_Path(dcm)))
        pos_pixels = np.sum(label > 0)
        attribs[KIDNEY_PIXELS] = pos_pixels
    else:
        attribs[KIDNEY_PIXELS] = None

    """
    Volume for pixels in mask = VOXEL_VOLUME * pos_pixels
    TKV calculated as summation of dcm volumes in a study
    Note: Dimension (which determines pixel-count) must be normal to calc. TKV
    """
    dX_Y = float(pdcm.PixelSpacing[0])
    dZ = None
    if 'SpacingBetweenSlices' in pdcm:
        dZ = float(pdcm.SpacingBetweenSlices)    
    elif 'SliceThickness' in pdcm:
        dZ = float(pdcm.SliceThickness)
    else:
        raise "dZ not available -- no SpacingBetweenSlices nor SliceThickness"

    attribs[VOXEL_VOLUME] = dZ * (dX_Y ** 2)
    attribs[DIMENSION] = arr_int16.shape

    return attribs


@functools.lru_cache()
def make_dcmdicts(dcms, label_status=True, WCM=True):
    """creates two dictionares with dcm attributes

    Arguments:
        dcms (tuple): tuple of dicoms. Note, tuple is used, rather than a list,
        so the input is hashable for LRU.

    Returns:
        dcm2attribs (dict), pt2dcm (dict):
            Dictionaries with dcms to attribs and patients to dcms
    """

    # convert tuple back to list
    if not isinstance(dcms, list):
        dcms = list(dcms)

    dcm2attribs = OrderedDict()
    patient2dcm = OrderedDict()

    exceptions = []
    for dcm in dcms:
        try:
            attribs = dcm_attributes(dcm, label_status, WCM=WCM)
            dcm2attribs[dcm] = attribs
            patient2dcm.setdefault(attribs[PATIENT], []).append(dcm)
        except Exception as e:
            exceptions.append(f"{e} with dcm:{dcm.name} ")

    if len(exceptions) > 0:
        print(
            "\n\nThe following exceptions were encountered: \n"
            f" {exceptions}\n\n"
        )

    return dcm2attribs, patient2dcm


# deprecated function
def filter_dcm2attribs(filters, dcm2attribs):
    """filters input dcm2attribs dict based on dict of filters
    (Note: Modifies input dcm2attribs)

    Arguments:
        filters {dict} -- dict of filters
            e.g. filters = {'seq':'AX SSFSE ABD/PEL'}
        dcm2attribs {dict} -- dict of dcms:
            attributes generated by function make_dcmdicts()
    Returns:
        dcm2attribs {dict} -- dict of dcms to attributes after filter
    """

    remove = []
    for dcm, attribs in dcm2attribs.items():
        for key, value in filters.items():
            if key not in attribs or value != attribs[key]:
                remove.append(dcm)

    for dcm in remove:
        del dcm2attribs[dcm]

    return dcm2attribs


def masks_to_colorimg(masks):
    """converts one hot encoded mask to color encoded image"""

    if np.ndim(masks) == 2:
        masks = np.expand_dims(masks, 0)

    # color codes for mask .png labels
    colors = [
        (201, 58, 64),  # Red
        (242, 207, 1),  # Yellow
        (0, 152, 75),  # Green
        (101, 172, 228),  # Blue
        (245, 203, 250),  # Pink
        (239, 159, 40),
    ]  # Orange

    colors = np.asarray(colors)[: masks.shape[0]]

    _, height, width = masks.shape
    colorimg = np.ones((height, width, 3), dtype=np.float32) * 255

    for y in range(height):
        for x in range(width):
            pixel_color = np.asarray(masks[:, y, x] > 0.5)
            selected_colors = colors[pixel_color]

            # assign pixels mean color RGB for display
            if len(selected_colors) > 0:
                colorimg[y, x, :] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)


def display_sample(sample):

    dcm, mask = sample

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(dcm, cmap="gray")
    ax2.imshow(dcm, cmap="gray")
    ax2.imshow(masks_to_colorimg(mask), alpha=0.5)


def display_verbose_sample(verbose_sample):

    (dcm, mask), path, attribs = verbose_sample

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(dcm, cmap="gray")
    ax2.imshow(dcm, cmap="gray")
    ax2.imshow(masks_to_colorimg(mask), alpha=0.5)

    print("\nPath: {}".format(path))
    print("\nAttribs: {}".format(attribs))


def display_traindata(inputs, labels):

    for index in range(0, inputs.shape[0]):
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(inputs[index][1], cmap="gray")

        axarr[1].imshow(inputs[index][1], cmap="gray")  # background for mask
        axarr[1].imshow(masks_to_colorimg(labels[index]), alpha=0.5)

        img = inputs[index][0]
        lb = masks_to_colorimg(labels[index])
