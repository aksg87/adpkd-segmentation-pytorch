# Python Daisy Chain
# Runs the daisy chain for arbitrary inferences
# Simpler iteration. Kidney-Liver-Spleen combination, softmax not yet incorporated
# By: Dom Romano
# Original: 09-14-2021
# Update:   02-03-2022
# For info on the update, please read the update notes text file.
# %% Import Functions
import glob
import json
import os
import subprocess
from pathlib import Path
from argparse import ArgumentParser
import nibabel as nib

from daisy_chain_utils import (
    comb_list,
    glob_extensions,
    path_slash,        
    inference_scan_glob,
    mask_scan_glob,
    select_plane_key,
    mask_add,
    organ_repaint,
    overlap_repaint,
    get_dicom_vol,
)
# %% Preliminary data
print("Configuring preliminary data...")
group = "pkd"
config_code = " --config_path " # Will probably load in from a json --> very similar properties to a dictionary
print("Loading model locations...")
id_config = open("/opt/pkd-data/akshay-code-2/daisy_chain/config_path.json",'r') # This is what gets placed into json.loads()
config_dict = json.loads(id_config.read())
akshay_dir_pre = "/opt/pkd-data/akshay-code-2/"  
akshay_dir_post = "-segmentation-pytorch"
input_code = " -i "
output_code = " -o "
activate_env = ". /opt/pkd-data/akshay-code/adpkd-segmentation-pytorch/adpkd_env_cuda_11_2/bin/activate" # Something is up with the cuda environment
#activate_env = ". /opt/pkd-data/akshay-code/adpkd-segmentation-pytorch/adpkd_env/bin/activate"
python_rel_cmd = "python adpkd_segmentation/inference/dom_inference_nocrop.py"
# akshay_path = r"/big_data2/apkd_segmentation/storage/output/saved_inference/adpkd-segmentation-pytorch/Analysis_2/"
# akshay_path IS A HARD-CODED DESTINATION. Perhaps smarter iterations will ask for a save path.                   
# Model Config. DO NOT MODIFY UNLESS A NEW MODEL IS ALREADY TRAINED
operating_system = os.name
folder_partition = path_slash(operating_system)
organ_name = ("kidney", "liver", "spleen")
model_name = ("adpkd", "liver", "spleen")
organ_color = (2,4,8)  # ITK-SNAP label colors
overlap = (6, 10, 12, 14)
recolor = (1, 2, 8, 8) # I may have to adjust the color/algorithm
kidney_colors = (1,2)  # --> recolor right to left (1 --> 2)
map_spleen = (8,3) # Repaint spleen for user ease
"""
  The idea: Liver usually paints over right kidney, however any
  kidney voxel overlapping with liver will be 6 (because the color
  is "red" for all kidneys before recoloring. These numbers provide
  unique overlaps, which is good for specific removal.
"""
youngest_child = "ITKSNAP_DCM_NIFTI"  # Last folder before dicom_vol and pred_vol nifti
dicom_suffix = '.dcm'
dicom_vol = "dicom_vol.nii"
pred_vol = "pred_vol.nii"
comb_folder_name = comb_list(organ_name)
combined_pred_filename = "comb_pred_vol.nii"
print("Preliminary Data Loaded (line 50)")
# %% Parser Setup
parser = ArgumentParser()
parser.add_argument(
    "-i",
    "--inference_path",
    type=str,
    help="path to input dicom data (replaces path in config file)",
    default=None,
)

parser.add_argument(
    "-o",
    "--output_path",
    type=str,
    help="path to output location",
    default=None,
)

args = parser.parse_args()

inference_path = args.inference_path
output_path = args.output_path
# %% Prep the output path 
if inference_path is not None:
    inf_path = inference_path

if output_path is not None:
    out_path = output_path
    if out_path[-1] == folder_partition:
        save_base = out_path  # String that we will use to flexibly build other directories
        temp_base = Path(save_base[0:-1])
        temp_base.mkdir(parents=True, exist_ok=True)        

    elif out_path[-1] != folder_partition:
        temp_base =Path(out_path)
        temp_base.mkdir(parents=True, exist_ok=True) # Makes the MRN_Date folder so we can populate with the organ files
        save_base = out_path + folder_partition   
        
print('Save Base Path:')
print(save_base)     
# %% Create Combined Parent Path
comb_parent_path = Path(temp_base) / comb_folder_name
comb_parent_path.mkdir(parents=True, exist_ok=True)
# %% Daisy Chain --> Runs for all organs ############################
pred_load_dir = []
scan_folders = inference_scan_glob(inf_path,dicom_suffix) # List me them dicoms pls
for organ in range(0,len(model_name)):
    print("Run " + str(organ + 1) + ": " + organ_name[organ] + " inference...\n")
    save_path = save_base + organ_name[organ]    
    pred_load_dir.append(save_path)  # The constructed inference save directory -> load    
    for scan_path in scan_folders:  
        print(scan_path)
        print('Matching the model to current patient orientation...')
        scan_dicoms = glob.glob(scan_path+folder_partition+'*'+dicom_suffix)
        plane_key = select_plane_key(scan_dicoms)
        print('Selected orientation model: ' + plane_key)        
        python_inf_dir = akshay_dir_pre + model_name[organ] + akshay_dir_post        
        cd_cmd = "cd " + python_inf_dir
        # will now be a load directory for the simple combination.
        run_python = python_rel_cmd + config_code + config_dict[plane_key][organ] + input_code + scan_path + output_code + save_path    
        full_command = activate_env + "; " + cd_cmd + "; " + run_python
        subprocess.call(full_command, shell=True) # This is the call to run the python code
        print(organ_name[organ] + " inference complete")


####################################################################
# %% Combine the Daisy Chain
print("Combining the organ segmentations...")
scan_list = glob_extensions(pred_load_dir[0], pred_vol)
if len(scan_list) == 1:
    print("One scan detected for this study. Processing...")
elif len(scan_list) > 1:
    print(str(len(scan_list)) + ' scans detected.')

for scan in range(0,len(scan_list)): # Even if there is only one scan, the single scan is in a list, so the loop would run once
    scan_folder, _ = mask_scan_glob(scan_list[scan], pred_vol, youngest_child) # This will copy the scan name. I will try this out with os.path.split(scan_list[scan])    
    print("Combining for " + scan_folder)
    overlap_mask = mask_add(pred_load_dir, organ_color, pred_vol, scan) ### UPGRADE
    print("Repainting overlaps...")
    recolor_spleen_comb_mask = overlap_repaint(overlap_mask, overlap, recolor, kidney_colors)
    print("Overlaps repainted. Repainting spleen...")
    comb_mask = organ_repaint(recolor_spleen_comb_mask, "all", map_spleen[0], map_spleen[1])
    print("Overlaps repainted.")
    # Constructing load path
    load_parent, _ = os.path.split(scan_list[scan]) # Detatch the mask file
    load_path = Path(load_parent)
    print("Loading MRI NIFTI and pulling necessary parameters...")
    mri_nifti, nifti_affine, nifti_header = get_dicom_vol(load_path / dicom_vol) # Attach the patient nifti
    comb_save_path = comb_parent_path / scan_folder
    comb_save_path.mkdir(parents=True, exist_ok=True)
    nib.save(mri_nifti, comb_save_path / dicom_vol)
    combined_pred_vol = nib.Nifti1Image(comb_mask, affine=nifti_affine, header=nifti_header)
    nib.save(combined_pred_vol, comb_save_path / combined_pred_filename)
    if scan != (len(scan_list)-1):
        print('Combined Prediction Mask Saved. Moving to next scan...\n')
    elif scan == (len(scan_list)-1):
        print('Combined Prediction Mask Saved for all scans.\n')
# %% Change the group
print("Changing Group Permissions...")
chgrp_cmd = "chgrp -R " + group + " " + save_base
subprocess.call(chgrp_cmd, shell=True) # This changes the permisions group of the output 
print("Group changed.")
print("Processing Complete. Please check the output files in the following path:")
print(comb_parent_path)
