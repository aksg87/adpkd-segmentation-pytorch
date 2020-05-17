#%%
import glob
import os
from pathlib import Path
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