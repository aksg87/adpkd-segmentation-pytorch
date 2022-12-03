from argparse import ArgumentParser
import os
from pathlib import Path
import yaml
import torch


from inference_utils import load_config, inference_to_nifti
from ensemble_utils import (
    scan_list,
    select_sequence_key,
    select_plane_key,
    binary_inference_to_disk,
    argmax_ensemble,
    ensemble_to_nifti,
)

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
    help="path to config file for inference pipeline",
    default="adpkd_segmentation/inference/ensemble_config.yml",
)


def run_binary_inference(
    config_list,
    organ_list,
    individual_flag,
    inference_path=None,
    output_path=None,
):
    # Run Inferences
    for i, organ in enumerate(organ_list):
        print(f"Running {organ}...")
        config = config_list[i]

        model_args = load_config(
            config_path=config, inference_path=inference_path
        )

        model_args.pop("model_name")

        if output_path is not None:
            model_args["save_dir"] = os.path.join(output_path, organ)

        binary_inference_to_disk(**model_args)
        torch.cuda.empty_cache()

        if individual_flag:
            print("Saving individual organ segmentation to NIFTI...")
            study_dir = os.path.join(output_path, organ)
            inference_to_nifti(inference_dir=study_dir)


def run_ensemble_inference(
    inference_path: str,
    output_binary_path: str,
    output_ensemble_path: str,
    config_path="adpkd_segmentation/inference/ensemble_config.yml",
):
    print("Loading system and pipeline configuration...")
    with open(config_path, "r") as id_system:
        try:
            system_config = yaml.load(id_system, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    # Set pointers to model yaml configs
    model_config_dict = system_config["model_dir"]
    binary_studies = []
    ensemble_studies = []
    dicoms = list(Path(inference_path).glob("**/*.dcm"))
    inference_folders, scans, example_dicoms = scan_list(
        dicoms, rules_dict=system_config["filename_rules"]
    )
    for inference_folder, scan, example_dicom in zip(
        inference_folders, scans, example_dicoms
    ):
        print("Selecting model...")
        sequence = select_sequence_key(example_dicom)
        orientation = select_plane_key(
            example_dicom,
            reference_directions=system_config["reference_planes"],
            plane_keys=system_config["plane_keys"],
        )
        config_list = model_config_dict[sequence][orientation]
        binary_study_dir = os.path.join(output_binary_path, scan)
        ensemble_dir = os.path.join(output_ensemble_path, scan)
        binary_studies.append(binary_study_dir)
        ensemble_studies.append(ensemble_dir)
        run_binary_inference(
            config_list=config_list,
            organ_list=system_config["organ_name"],
            individual_flag=system_config["individual_flag"],
            inference_path=inference_folder,
            output_path=binary_study_dir,
        )

    print("Ensembling multi-organ segmentations...")
    argmax_ensemble(
        scan_list=binary_studies,
        output_folder=output_ensemble_path,
        organ_name=system_config["organ_name"],
        index_classes=system_config["ensemble_index"],
        itk_colors=system_config["inference_ensemble_color"],
    )
    print("Creating Ensemble nifti...")
    ensemble_to_nifti(
        output_scan_list=ensemble_studies,
        selected_kidney_side=system_config["kidney_side"],
        kidney_ensemble_color=system_config["kidney_ensemble_color"],
        kidney_side_color=system_config["kidney_side_color"],
    )


if __name__ == "__main__":
    args = parser.parse_args()
    inference_path = args.inference_path
    output_path = args.output_path
    output_binary_path = os.path.join(output_path, "single_organ_inference")
    output_ensemble_path = os.path.join(output_path, "multi_organ_ensemble")
    config_path = args.config_path

    run_ensemble_inference(
        inference_path=inference_path,
        output_binary_path=output_binary_path,
        output_ensemble_path=output_ensemble_path,
        config_path=config_path,
    )
    # TODO: Test this entire pipeline!
