from pathlib import Path

# adapt the file for actual data locations

dataroot = Path("data")
labeled_dirs = [dataroot / "processed"]

unlabeled_dirs = [
    dataroot / "unlabelled_data"
]  # unused currently for purely supervised training approach

print(f"making links for {labeled_dirs}")

# modify for custom symbolic link locations
LABELED = None
UNLABELED = None

# default location
script_location = Path(__file__).resolve()

if LABELED is None:
    LABELED = script_location.parent.parent.parent / "labeled"
if UNLABELED is None:
    UNLABELED = script_location.parent.parent.parent / "unlabeled"
