from pathlib import Path
from argparse import ArgumentParser

from adpkd_segmentation.inference.inference_utils import load_config

from adpkd_segmentation.evaluation.evaluation_utils import (
    inference_to_disk,
    compute_inference_stats,
)

parser = ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    help="config_path for inference",
    default=(
        "experiments/runs/best_experiments/"
        "2_new_stratified_run_2_long_advprop_640_batch_dice_1/val/val.yaml"
    ),
)


def run_evalutation_stats(
    config_path,
    SAVED_INFERENCE="saved_inference",
    SAVED_FIGS="saved_figs",
):

    # %%
    # Run inferences
    model_args = load_config(config_path=config_path)

    # load_config initializes all objects including:
    # model and datloader for InferenceDataset

    inference_to_disk(*model_args)

    # %%
    compute_inference_stats(
        save_dir="./saved_inference", display=True, output=True
    )

    # TODO: Add plots


if __name__ == "__main__":
    args = parser.parse_args()
    print("args ", args)
    config_path = Path(args.config_path)
    run_evalutation_stats(
        config_path=config_path,
    )
