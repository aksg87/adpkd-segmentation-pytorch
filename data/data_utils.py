#%%
import glob
import os

#%%

def get_dcms_paths(dir_list):

    all_files = []

    for dir in dir_list:
        print("processing {} ".format(dir))
        
        files = glob.glob('{}/**/*.dcm'.format(dir), recursive=True)
        all_files.extend(files)

        print("total files... --> {} \n".format(len(all_files)))

    return all_files