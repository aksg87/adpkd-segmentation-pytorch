#%%
import glob
import os
from pathlib import Path
import numpy as np
#%%

def get_dcms_paths(dir_list):

    all_files = []

    for dir in dir_list:
        print("processing {} ".format(dir))
        
        files = glob.glob('{}/**/*.dcm'.format(dir), recursive=True)
        all_files.extend(files)

        print("total files... --> {} \n".format(len(all_files)))

    return all_files

def get_y_Path(x):
    """Get label path from dicom path"""    

    if isinstance(x, str):
        x = Path(x)

    y = str(x.absolute()).replace('DICOM_anon', 'Ground')
    y = y.replace('.dcm', '.png')    
    y = Path(y)

    return y

def mask2label(mask, data_set_name="ADPKD"):
    """converts mask png to one-hot-encoded label"""    

    #unique_vals corespond to mask class values after transforms        
    if(data_set_name == "ADPKD"):
        L_KIDNEY = 0.5019608
        R_KIDNEY = 0.7490196
        unique_vals = [R_KIDNEY, L_KIDNEY]

    mask = mask.squeeze()
    
    s = mask.shape

    ones, zeros = np.ones(s), np.zeros(s)

    one_hot_map = [np.where(mask == unique_vals[targ], ones, zeros)
                   for targ in range(len(unique_vals))]

    one_hot_map = np.stack(one_hot_map, axis=0).astype(np.uint8)

    return one_hot_map

def masks_to_colorimg(masks):
    """converts mask png grayscale to color encoded image""" 

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