#%%
import os
import glob
import ntpath  
import shutil

from .data_utils import get_dcms_paths, get_y_Path, make_dcmdicts, get_labeled, get_labeled
from .data_config import labeled_dirs, unlabeled_dirs # define data sources in data_config.py
#%%
def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except Exception as e:
        print(e)

def mkdir_force(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
# %%
def makelinks():

    mkdir_force('labeled')
    mkdir_force('unlabeled')

    for dcm in get_dcms_paths(labeled_dirs):
        mask = get_y_Path(dcm)
        if not mask.exists():
            raise Exception('Labeled dcm [{}] does not have mask.'.format(dcm))

        symlink_force(dcm, 'labeled/'+ os.path.basename(dcm))
        symlink_force(mask, 'labeled/'+ os.path.basename(mask))

    for dcm in get_dcms_paths(unlabeled_dirs):
        mask = get_y_Path(dcm)

        if mask.exists():
            raise Exception('Unlabeled dcm [{}] contains a mask.'.format(dcm))

        symlink_force(dcm, 'unlabeled/'+ os.path.basename(dcm))