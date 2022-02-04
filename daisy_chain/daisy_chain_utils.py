## Daisy Chain Utils
"""
    This will be any utility functions needed to build or test the daisy chain functionality of the end product.
"""
# Import relevant modules
import glob
import os
from pathlib import Path
import subprocess
import numpy as np
import pydicom
import nibabel as nib
# Functions
############Slated For Removal (Replaced by Globbing Methods)############
# %% Function: list_folders. Original Author: Jinwei Zhang
def list_folders(rootDir = '.', sort=0):

    if not sort:
        return [os.path.join(rootDir, filename)
            for filename in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, filename))]
    else:
        return [os.path.join(rootDir, filename) for filename in sorted(os.listdir(rootDir))]


# %% An extension of list folers, this will return any directories with the appropriate suffix. I think I can even improve my daisy chain code with this.
"""
    So the main idea here is that I will search all outputs for two things:
    1) dicom_vol.nii and pred_vol.nii (just a sanity check. I will allow this to be one case)
    2) logits and dicoms (the logits needed to be loaded in if I stack it up)
"""
def list_files_with_suffix(rootDir = '.', suffix = ''):

    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootDir)
        for filename in filenames if filename.endswith(suffix)]
        
        
# %% List parent path of all folders (If I EVER Need it)
def list_parent_of_file(rootDir = '.', suffix = ''):
    a = [looproot
        for looproot, _, filenames in os.walk(rootDir)
        for filename in filenames if filename.endswith(suffix)] # This is NOT the most ideal solution

    return a # This is a little different compared to the boundary_utils functioin grap_parent_of_file (that is a single output, not a list)


#########################################################################
############## Globbing Methods ##############
# %% Get files with a specific extention
def glob_extensions(ref_path,file_ext):
    """Recursively return paths with the desired
    file extension. Slightly more concise way of obtaining the
    files compared to 'list_files_with_suffix'"""
    slash = path_slash(os.name) # Will give methe slash I want
    recursive_path = ref_path + slash + '**' + slash + '*' + file_ext 
    # '/**/*' (posix) '\**\*' (windows) specifies recursive searches
    return glob.glob(recursive_path, recursive=True)

    
def parent_glob(dir_list):
    """For a given list of files, find the parent folder
    input: dir_list: list of files
    output: list of parent paths from os.path.dirname"""
    return [os.path.dirname(directory) for directory in dir_list]


def unique(input_list):
    """For a given list with arbitrary elements,
    return a list that only contains the unique elements
    of the input"""
    # Initialize a null list
    unique_list = []
    # Traverse for all elements
    for x in input_list:
        # check if this already exists
        if x not in unique_list:
            unique_list.append(x)

    return unique_list


def inference_scan_glob(ref_path, file_ext):
    """This puts together the above three functions:
    1) give me all of the files of given extension 'file_ext' 
      in the given path 'ref_path'
    2) give me the the parent directories of the files
    3) return the unique parent directories."""
    all_files = glob_extensions(ref_path, file_ext)    
    parents = parent_glob(all_files)    
    return unique(parents)


# %% Scan from glob
def mask_scan_glob(input_dir,omit_file,omit_folder):
    """Instead of determining the scan from dicom globbing, we are 
    finding the scan folder of the output that holds the mask.
    This is more custom written for the inference pipeline than
    it is for an arbitrary path."""
    temp_head, temp_folder = os.path.split(input_dir)
    if temp_folder == omit_file:
        temp_head, temp_folder = os.path.split(temp_head)
        if temp_folder == omit_folder:
            _, temp_folder = os.path.split(temp_head)
    return temp_folder, temp_head


#########################################################################
# %% Combine filenames
def comb_list(name_list):
    if len(name_list) <= 3 and len(name_list) > 1:
        temp_name = ""
        for name in range(0,len(name_list)):
            temp_name += "_" + name_list[name]
        comb_name = "Combined" + temp_name
    elif len(name_list) > 3:
        comb_name = "Combined"
    return comb_name
    

# %% Return the maximum index
def max_ind(input_list):
    """Given the input of a list, return the index at which
    the maximum occurs. This works best as long as you know
    you aren't feeding in a constant list. You can verify this 
    with np.unique()
    """
    element_max = np.max(input_list)
    for index in range(len(input_list)):
        if input_list[index] == element_max:
            return index


################ Plane Selection Functions ##############################
def select_plane(input_dicom):
    """Selects the plane from the 'patient orientation' DICOM 
    attribute. This should be the most robust way of doing so. 
    The main idea is this: We get the coordinates of direction 
    cosines from the attribute, and we KNOW what the 'axial', 
    'coronal' and 'saggital' directional cosine components
    So then we can compare our patient orientation with the 
    known orientations via an inner product. The inner product
    closest to one (or the maximum in this case) will 
    be the most likely orientation.
    Basic implementation:
        1) The axial, coronal, and saggital orientations are
            pre-defined by people smarter than me
        2) Read the 'real-world' patient data and convert their
            image orientation into a usable numpy array
        3a) Calculate the reference basis direction
        3b) Calclate the patient direction
        4) Compare the reference and patient directions
            i) First take the dot (inner) product between 
                the reference and patient directions for
                each reference (Ax, Cor, Sag)
            ii) Return the position of the largest dot product
    **: Note that highly oblique edge cases (\phi, \theta \to \pi/4)
        were NOT considered here (equalities would occur in this 
        case).
        -"""
    plane_vecs = np.matrix('1 0 0 0 1 0; 1 0 0 0 0 -1; 0 1 0 0 0 1')
    V = pydicom.dcmread(input_dicom)
    dicom_orientation = V[0x00200037].value
    ori_vec = np.array(dicom_orientation)
    patient_direction = np.cross(ori_vec[0:3],ori_vec[3:6]) # Calculates the cartesian direction of the patient plane
    dot_ori = np.zeros(np.shape(plane_vecs)[0])
    for ind in range(np.shape(plane_vecs)[0]):
        ref_basis = np.cross(plane_vecs[ind,0:3], plane_vecs[ind,3:6]).flatten() # Reference direction (cartesian basis)                
        dot_ori[ind] = np.absolute(np.dot(patient_direction, ref_basis))        
        # The above line is returns the projection of the patient direction onto the the reference directions 
        # I only care if the vectors are parallel, which is why I am doing |<u,v>|/|v|
    print("Maximum dot product of orientations: " + str(dot_ori[max_ind(dot_ori)]))
    return max_ind(dot_ori) # The largest inner product will give the best model by axial plane


def select_plane_key(dicom_list): 
    """For a given input of dicoms, select the key that 
    corresponds to the scan plane.This assumes that the
    first dicom in the series has the orientation I am 
    looking for."""
    if type(dicom_list) == list: # Even if the elements are strings, the list characteristic overrides!
        temp_dicom = dicom_list[0] 
    elif type(dicom_list) == str: 
        temp_dicom = dicom_list

    key_list = ['Axial', 'Coronal', 'Sagittal'] # CONFIRM WITH THE CONFIG FILE
    plane_index = select_plane(temp_dicom)      
    return key_list[plane_index] # The key for the imaging plane 


#########################################################################
# OS paths
# %% I think this is  a silly warning coming form the code. I can try testing this in RadDeep and debugging there if worst comes to worst
def path_slash(name):
    
    """
        Just use os.name as the input to find the output the following folder
        identifiers based on the operating system.
        Windows: backslash             
        Unix: "/"
    """

    if name == 'nt':
        return "\\"
    elif name == 'posix':
        return "/"


# %% Get the folder from a file
def get_folder(root_dir = '.'):
    _, folder = os.path.split(root_dir)
    
    return folder


# %% Get to the scan directories 
def get_scan_paths(root_dir = '.', target_file = 'dicom_vol.nii', remove_parent = 'ITKSNAP_DCM_NIFTI'):

    """
        For a known target file that is uniquely saved in each scan folder, this function will
        return the list of available scans inferred AND will return the patient
        MRN, which may be useful depending on which data I want.
        If no child number is given, then this will default an output to one child path.
    """

    temp_dir = glob_extensions(root_dir, target_file)
    scan_path = []
    for scan_dir in temp_dir:
        temp_head, temp_path = mask_scan_glob(scan_dir,target_file, remove_parent)
        scan_path.append(temp_path)
    
    return scan_path                     


# %% Let's just make a quick load path function
def get_load_dir(load_path, desired_file, scan = 0, rel_folder = "ITKSNAP_DCM_NIFTI"):
    list_load = get_scan_paths(load_path, desired_file)
    load_organ_scan = list_load[scan] # Ensures consistency with the main script, even if I have to add two lines
    load_path = Path(load_organ_scan) / rel_folder
    load_dir = load_path / desired_file
    return load_dir


# %% Initialize numpy from load_path
def init_numpy_arr(load_path, target_file = "dicom_vol.nii", scan = 0):
    load_dir = get_load_dir(load_path, target_file, scan)    
    temp_pred_vol_nii = nib.load(load_dir)
    temp_pred = np.array(temp_pred_vol_nii.dataobj)    
    temp_init = np.zeros(np.shape(temp_pred))
    return temp_init


# %% Affine and header parameters
def pull_nifti_params(load_dir, target_file = "dicom_vol.nii", scan = 0):
    load_dir = get_load_dir(load_dir, target_file, scan)    
    temp_dicom_nii = nib.load(load_dir)
    pred_affine = temp_dicom_nii.affine
    pred_header = temp_dicom_nii.header.copy()
    return pred_affine, pred_header
    

# %% Copy path and change group
def copy_command(copy_path, destination, folder_name, group):
    path_dest = Path(destination)
    organ_folder = get_folder(copy_path)
    path_dest.mkdir(parents=True, exist_ok=True)
    print("Copying and changing group permissions...")
    rename_command = "mv " + destination + organ_folder + " " + destination + folder_name    
    copy_command = "cp -r " + copy_path + " " + destination
    chgrp_cmd = "chgrp -R " + group + " " + destination + folder_name
    full_command = copy_command + "; " + rename_command + "; " + chgrp_cmd
    subprocess.call(full_command, shell=True)        


# %% Find the midline of an np array
def find_mid_ind(img,dim):
    """
    img - Input image, or multidimensional numpy array
    dim - the dimension of the numpy array shape
    Output: the middle coordinate of the image.
    """
    img_shape = np.shape(img)
    if dim >= 0 and dim <= (len(img_shape) - 1):
        if img_shape[dim] % 2 == 0: # If the size is even
            mid_ind = 0.5*img_shape[dim] - 1 # The -1 is to account for the fact that we start at 0
        elif img_shape[dim] % 2 == 1:
            mid_ind = 0.5*(img_shape[dim] + 1) - 1 # It will move the midline index as the index of symmetry            
    else:
        print('The inserted dimension is ' + str(dim))
        print('The inserted dimension is not admittable. Plsease enter a dimension between [0,N-1]')
        mid_ind = []  # This will intentionally throw an error
    
    return np.uint(mid_ind)

# %% Repaint the kidney based on desired color AND anatomical side
def organ_repaint(mask,side,orig_color,new_color):
    x_mid_ind = find_mid_ind(mask,0)  # I understand this is hard coded for right now. this is the image x-axis
    N = np.shape(mask)
    x_ind = np.arange(0,N[0])
    if side == "right" or side == "Right" or side == "r" or side == "R" or side == 0:
        right_half = np.int16(mask[x_ind < x_mid_ind, :, :])    
        right_half[right_half == orig_color] = new_color
        mask[x_ind < x_mid_ind, :, :] = right_half
    elif side == "all" or side == "ALL" or side == "fov" or side == "FOV":
      full_fov = np.int16(mask)
      full_fov[mask == orig_color] = new_color
      mask = full_fov
    else:
        left_half = np.int16(mask[x_ind > x_mid_ind, :, :])
        left_half[left_half == orig_color] = new_color
        mask[x_ind > x_mid_ind, :, :] = left_half
    #
    return np.uint16(mask)


# %% Overlap Repaint Function
"""
    This will be an extra line in the daisy chain code. Right after mask_add since I don't want to have too many input arguments
"""
def overlap_repaint(mask_arr,overlap_color,repaint_color, specify_organ=[]):
    temp_arr = mask_arr
    for color_ind in range(0, len(overlap_color)):
        temp_arr[mask_arr == overlap_color[color_ind]] = repaint_color[color_ind]

    if specify_organ == []:
        return np.uint16(temp_arr)
    else:
        adjusted_arr = organ_repaint(temp_arr, "left", specify_organ[0], specify_organ[1])  # THIS is a hot fix for the current overlap problem.
        # TODO: Extend this functionality if possible.
        return np.uint16(adjusted_arr)


# %% Simple mask combination
def mask_add(load_list, mask_list, filename_mask, scan = 0, side = "right"): 
    temp_combine = init_numpy_arr(load_list[0], filename_mask, scan)   #  UPGRADE!!!!! Init from the first predictied nifti from akshay inference and the required scan so the dimensions match up    
    #pred_affine, pred_header = pull_nifti_params(load_list[0], filename_mask, scan) # UPGRADE!!!!! Correct scan number will have the correct header info
    for mask in range(0,len(mask_list)):                
        load_dir = get_load_dir(load_list[mask], filename_mask, scan)  # UPGRADE!!!! 
        temp_pred_vol_nii = nib.load(load_dir)
        temp_pred = np.array(temp_pred_vol_nii.dataobj)        
        temp_combine += mask_list[mask]*temp_pred        
        # if mask == (len(mask_list) - 1):
                
    
    # temp_int = np.uint16(temp_combine)
    temp_int = organ_repaint(temp_combine, side, mask_list[0], 1) # mask_list[0] pertains to the overall kidney seg
    return temp_int
    # combined_pred_vol = nib.Nifti1Image(temp_int, affine=pred_affine, header=pred_header)         
    # nib.save(combined_pred_vol, save_dir)


# %% Retrieve the relevant diocm file
def get_dicom_vol(load_dir):

    """
        For a given load directory and file name, this will return the 
        dicom -> nifti scan, header, and affine
            mri_nifti -- dicom -> nifti scan
            nifti_header -- the copied header of the scan
            nifti_affine -- the affine of the scan. The segmentation
            and mri_nifti affines must match for things to make sense.
    """

    mri_nifti = nib.load(load_dir)
    nifti_affine = mri_nifti.affine
    nifti_header = mri_nifti.header.copy()    
    return mri_nifti, nifti_affine, nifti_header
    