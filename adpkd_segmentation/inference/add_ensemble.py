import json
from argparse import ArgumentParser
import os
from pathlib import Path
import nibabel as nib

from inference import run_inference
from ensemble_utils import get_scan, addition_ensemble


# Parser Setup
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

parser.add_argument(
    "-c",
    "--config_path",
    type=str,
    help="Path that points to the desired configuration file",
    default="adpkd_segmentation/inference/ensemble_config.json",
)


def run_addition_ensemble(
    input_path=None,
    output_path=None,
    config_path=None,
):
    # Preliminary Data
    print("Loading system and pipeline configuration...")
    with open(config_path, "r") as id_system:
        system_config = json.loads(id_system.read())
    # Individual Organ inference
    pred_load_dir = []
    for idx_organ, name_organ in enumerate(system_config["organ_name"]):
        print(
            "Run " + str(idx_organ + 1) + ": " + name_organ + " inference...\n"
        )
        save_path = os.path.join(output_path, name_organ)
        config_path = system_config["model_dir"]["T2"]["Axial"][idx_organ]
        saved_inference = Path(save_path) / system_config["svd_inf"]
        saved_figs = Path(save_path) / system_config["svd_figs"]
        run_inference(
            config_path=config_path,
            inference_path=input_path,
            saved_inference=saved_inference,
            saved_figs=saved_figs,
        )
        pred_load_dir.append(Path(save_path))
        print(name_organ + " inference complete")
        #
    # Create ensemble save path
    temp_name = ""
    if len(system_config["organ_name"]) <= system_config["max_organ_title"]:
        for name in system_config["organ_name"]:
            temp_name += f"_{name}"
    else:
        temp_name = "_organs"
    combined_folder_name = f"Addition_Ensemble{temp_name}"
    combine_path = Path(output_path) / combined_folder_name
    # Addition Ensemble
    print("Combining the organ segmentations...")
    scan_list = list(pred_load_dir[0].glob(f'**/*{system_config["pred_vol"]}'))
    for idScn, scan in enumerate(scan_list):
        temp_dict = {"organ_paths": pred_load_dir, "idS": idScn}
        scan_folder = get_scan(system_config, scan)
        print(f"Combining for scan {scan_folder}")
        comb_mask = addition_ensemble(temp_dict, system_config)

        # Save the output
        save_path = combine_path / scan_folder
        save_path.mkdir(parents=True, exist_ok=True)
        #
        dicom_parent, _ = os.path.split(scan)
        dicom_nii_dir = Path(dicom_parent) / system_config["dicom_vol"]
        mri_nifti = nib.load(dicom_nii_dir)
        nifti_affine = mri_nifti.affine
        nifti_header = mri_nifti.header.copy()
        combined_pred_vol = nib.Nifti1Image(
            comb_mask, affine=nifti_affine, header=nifti_header
        )
        print("Saving...")
        nib.save(mri_nifti, save_path / system_config["dicom_vol"])
        nib.save(
            combined_pred_vol,
            save_path / system_config["combined_pred_filename"],
        )


if __name__ == "__main__":
    args = parser.parse_args()

    inference_path = args.inference_path
    output_path = args.output_path
    config_path = "adpkd_segmentation/inference/ensemble_config.json"
    # Prep the output path
    if inference_path is not None:
        inf_path = inference_path

    if output_path is not None:
        out_path = output_path

    run_addition_ensemble(
        input_path=inf_path, output_path=out_path, config_path=config_path
    )
