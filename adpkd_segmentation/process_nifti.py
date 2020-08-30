"""Process nifti annotations to png masks"""

from argparse import ArgumentParser
from pathlib import Path

from adpkd_segmentation.utils.nifti_utils import process_nifti_dirs

# "source" can be any nested directory structure containing study directories
# each study directory should have a unique name, and it should
# contain "DICOM_anon" dir containing dcm files and "Untitled.nii.gz"
# check nifti_utils for expected constant definitions

# all study dirs will be processed and copied to a single root target dir
parser = ArgumentParser()
parser.add_arguments(
    "source", type=str, help="Source directory with annotations"
)
parser.add_arguments(
    "target", type=str, help="Target directory for processed studies"
)

if __name__ == "__main__":
    args = parser.parse_args()
    source_dir = Path(args.source)
    target_dir = Path(args.source)
    process_nifti_dirs(source_dir, target_dir)
