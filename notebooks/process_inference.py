# %%
# process studies into descriptive directories

from pathlib import Path
import pydicom
import os
import shutil

dir = Path("select_two/OUTSIDE")
out_dir = Path("select_two/processed")
files = list(dir.glob("**/*"))
print(len(files))
# %%

for f in files:
    try:
        d = pydicom.read_file(f)
        os.makedirs(
            str(out_dir / d.PatientID / d.SeriesDescription), exist_ok=True
        )
        print(d.PatientID)
        shutil.copy(
            f, out_dir / d.PatientID / d.SeriesDescription / f"{f.name}.dcm"
        )
    except:
        pass

# %%
