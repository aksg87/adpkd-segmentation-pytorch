import json
from argparse import ArgumentParser
import os
from pathlib import Path

from inference import run_inference


# Preliminary Data
print("Loading system and pipeline configuration...")
id_system = open("adpkd_segmentation/inference/ensemble_config.json", "r")
system_config = json.loads(id_system.read())
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


def run_addition_ensemble(input_path=None, output_path=None):
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
        )  # Test this tomorrow
        pred_load_dir.append(save_path)
        print(system_config["organ_name"][idx_organ] + " inference complete")
        #
        # Addition Ensemble -- Add in v1.02

    # Save the output -- Add in v1.02 or v1.03


if __name__ == "__main__":
    args = parser.parse_args()

    inference_path = args.inference_path
    output_path = args.output_path
    # Prep the output path
    if inference_path is not None:
        inf_path = inference_path

    if output_path is not None:
        out_path = output_path

    run_addition_ensemble(input_path=inf_path, output_path=out_path)
