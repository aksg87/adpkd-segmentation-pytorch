#%%
import glob
import os
from pathlib import Path
import numpy as np
import pydicom
from PIL import Image
from collections import defaultdict
import functools
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#%%

def get_dcms_paths(dir_list):

    all_files = []

    for dir in dir_list:
        print("processing {} ".format(dir))
        
        files = glob.glob('{}/**/*.dcm'.format(dir), recursive=True)
        all_files.extend(files)

        print("total files... --> {} \n".format(len(all_files)))

    return all_files

def get_labeled():
    dcms = glob.glob('{}/*.dcm'.format('labeled'))
    return dcms

def get_unlabeled():
    dcms = glob.glob('{}/*.dcm'.format('unlabeled'))
    return dcms

def get_y_Path(x):
    """Get label path from dicom path"""    

    if isinstance(x, str):
        x = Path(x)

    y = str(x.absolute()).replace('DICOM_anon', 'Ground')
    y = y.replace('.dcm', '.png')    
    y = Path(y)

    return y

def path_2dcm(fname):

    if not isinstance(fname, str):
        fname = str(fname)
    dcm = pydicom.dcmread(fname)
    return dcm.pixel_array.astype(dtype=np.float32)

def path_2label(fname):

    if not isinstance(fname, str):
        fname = str(fname)
    label = Image.open(fname)
    return np.array(label)

def dcm_attributes(dcm):

    attribs = {}

    # dicom header attribs
    pdcm = pydicom.dcmread(dcm)
    attribs["patient"] = pdcm.PatientID[:-3]
    attribs["MR"] = pdcm.PatientID[-3:]
    attribs["seq"] = pdcm.SeriesDescription

    # pixels in mask --> kidney
    label = np.array(Image.open(get_y_Path(dcm)))
    pos_pixels = np.sum(label>0)
    attribs["kidney_pixels"] = pos_pixels

    return attribs

@functools.lru_cache()
def make_dcmdicts(dcms):
    """creates two dictionares with dcm attributes

    Arguments:
        dcms (tuple): tuple of dicoms. Note, tuple is used, rather than a string, so the input is hashable for LRU.

    Returns:
        dcm2attribs (dict), pt2dcm (dict): Dictionaries with dcms to attribs and patients to dcms
    """     

    # convert tuple back to list 
    if not isinstance(dcms, list):
        dcms = list(dcms)

    dcm2attribs = defaultdict(tuple)
    patient2dcm = defaultdict(list)

    for dcm in dcms:
        attribs = dcm_attributes(dcm)
        dcm2attribs[dcm] = attribs
        patient2dcm[attribs["patient"]].append(dcm)
        
    return dcm2attribs, patient2dcm

def masks_to_colorimg(masks):
    """converts mask png grayscale to color encoded image""" 

    if np.ndim(masks) == 2:
        masks = np.expand_dims(masks, 0)

    # color codes for mask .png labels
    colors = [(201, 58, 64), #Red
            (242, 207, 1), #Yellow
            (0, 152, 75), #Green
            (101, 172, 228), #Blue
            (245, 203, 250), #Pink
            (239, 159, 40)] #Orange

    colors = np.asarray(colors) [:masks.shape[0]]

    _, height, width = masks.shape
    colorimg = np.ones((height, width, 3), dtype=np.float32) * 255

    for y in range(height):
        for x in range(width):
            pixel_color = np.asarray(masks[:, y, x] > 0.5)
            selected_colors = colors[pixel_color]

            #assign pixels mean color RGB for display
            if len(selected_colors) > 0:
                colorimg[y, x, :] = np.mean(selected_colors, axis=0)
            
    return colorimg.astype(np.uint8)


def display_sample(sample):

    dcm, mask = sample

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(dcm, cmap='gray')
    ax2.imshow(dcm, cmap='gray')
    ax2.imshow(masks_to_colorimg(mask), alpha=0.5)

def display_verbose_sample(verbose_sample):

    (dcm, mask), path, attribs = verbose_sample

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(dcm, cmap='gray')
    ax2.imshow(dcm, cmap='gray')
    ax2.imshow(masks_to_colorimg(mask), alpha=0.5)

    print("\nPath: {}".format(path))
    print("\nAttribs: {}".format(attribs))

def display_traindata(inputs, labels):

    for index in range(0, inputs.shape[0]):
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(inputs[index][1], cmap='gray')
        
        axarr[1].imshow(inputs[index][1], cmap='gray')  # background for mask
        axarr[1].imshow(masks_to_colorimg(labels[index]), alpha=0.5)

        img = inputs[index][0]
        lb = masks_to_colorimg(labels[index])