from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

from inference_utils import (
    load_config,
    inference_to_disk,
    display_volumes,
    inference_to_nifti,
)

parser = ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    help="config_path for inference",
    default="checkpoints/inference.yml",
)


def run_inference(
    config_path="checkpoints/inference.yml",
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
    # Creating figures for all inferences

    # Get all model inferences
    inference_files = Path(SAVED_INFERENCE).glob("**/*")

    # Folders are of form 'saved_inference/adpkd-segmentation/{PATIENT-ID}/{SERIES}'
    folders = [f.parent for f in inference_files if len(f.parent.parts) == 4]
    folders = list(set(folders))

    IDX_series = -1
    IDX_ID = -2

    saved_folders = [
        Path(SAVED_FIGS) / f"{d.parts[IDX_ID]}_{d.parts[IDX_series]}"
        for d in folders
    ]
    # %%
    # Generate figures for all inferences
    for study_dir, save_dir in tqdm(list(zip(folders, saved_folders))):
        try:
            # Save inference figure to save_dir
            display_volumes(
                study_dir=study_dir,
                style="pred",
                plot_error=True,
                skip_display=False,
                save_dir=save_dir,
            )

            inference_to_nifti(inference_dir=study_dir)

        except Exception as e:
            print(e)


if __name__ == "__main__":
    args = parser.parse_args()
    print("args ", args)
    config_path = Path(args.config_path)
    run_inference(
        config_path=config_path,
    )
