"""
Model evaluation script for TKV

python -m evaluate_patients --config path_to_config_yaml --makelinks

If using a specific GPU (e.g. device 2):
CUDA_VISIBLE_DEVICES=2 python -m evaluate_patients --config path_to_config_yaml

The makelinks flag is needed only once to create symbolic links to the data.
"""

# %%
from collections import OrderedDict, defaultdict
import argparse

import yaml
import pandas as pd

import torch

from config.config_utils import get_object_instance
from data.link_data import makelinks
from train_utils import load_model_data


# %%
def calculate_dcm_voxel_volumes(
    dataloader,
    model,
    device,
    binarize_func,
):
    num_examples = 0
    dataset = dataloader.dataset
    updated_dcm2attribs = {}

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        batch_size = y_batch.size(0)
        num_examples += batch_size
        with torch.no_grad():
            y_batch_hat = model(x_batch)
            y_batch_hat_binary = binarize_func(y_batch_hat)
            start_idx = num_examples - batch_size
            end_idx = num_examples

            for inbatch_idx, dataset_idx in enumerate(
                range(start_idx, end_idx)
            ):
                # calculate TKV and TKV inputs for each dcm
                # TODO:
                # support 3 channel setups where ones could mean background
                # needs mask standardization to single channel
                _, dcm_path, attribs = dataset.get_verbose(dataset_idx)
                attribs["pred_kidney_pixels"] = torch.sum(
                    y_batch_hat_binary[inbatch_idx] > 0
                ).item()
                attribs["ground_kidney_pixels"] = torch.sum(
                    y_batch[inbatch_idx] > 0).item()

                # TODO: Clean up method of accessing Resize transform
                attribs["transform_resize_dim"] = (
                    dataloader.dataset.augmentation[0].height,
                    dataloader.dataset.augmentation[0].width,
                )

                # scale factor takes into account the difference
                # between the original image/mask size and the size
                # after mask & prediction resizing
                scale_factor = (attribs["dim"][0] ** 2) / (
                    attribs["transform_resize_dim"][0] ** 2
                )
                attribs["Vol_GT"] = (
                    scale_factor *
                    attribs["vox_vol"] *
                    attribs["ground_kidney_pixels"]
                )
                attribs["Vol_Pred"] = (
                    scale_factor
                    * attribs["vox_vol"]
                    * attribs["pred_kidney_pixels"]
                )

                updated_dcm2attribs[dcm_path] = attribs

    return updated_dcm2attribs


# %%
def evaluate(config):
    model_config = config["_MODEL_CONFIG"]
    loader_to_eval = config["_LOADER_TO_EVAL"]
    dataloader_config = config[loader_to_eval]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]
    saved_checkpoint = config["_MODEL_CHECKPOINT"]
    checkpoint_format = config["_NEW_CKP_FORMAT"]

    model = get_object_instance(model_config)()
    if saved_checkpoint is not None:
        load_model_data(saved_checkpoint, model, new_format=checkpoint_format)

    dataloader = get_object_instance(dataloader_config)()

    # TODO: Hardcoded to dice_metric, so support other metrics from config.
    binarize_func = get_object_instance(
        loss_metric_config["criterions_dict"]["dice_metric"]["binarize_func"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    updated_dcm2attribs = calculate_dcm_voxel_volumes(
        dataloader, model, device, binarize_func
    )

    return updated_dcm2attribs


# %%
def calculate_TKVs(run_makelinks=False, output=None, path=None):
    if run_makelinks:
        makelinks()
    if path is None:
        path = "./example_experiment/train_example_all_no_noise_patient_seq_norm_b5_BN/val/val.yaml" # noqa
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # val or test
    split = config["_LOADER_TO_EVAL"].split("_")[0].lower()

    dcm2attrib = evaluate(config)

    patient_MR_TKV = defaultdict(float)
    TKV_data = OrderedDict()

    for key, value in dcm2attrib.items():
        patient_MR = value["patient"] + value["MR"]
        patient_MR_TKV[(patient_MR, "GT")] += value["Vol_GT"]
        patient_MR_TKV[(patient_MR, "Pred")] += value["Vol_Pred"]

    for key, value in dcm2attrib.items():
        patient_MR = value["patient"] + value["MR"]

        if patient_MR not in TKV_data:

            summary = {
                "TKV_GT": patient_MR_TKV[(patient_MR, "GT")],
                "TKV_Pred": patient_MR_TKV[(patient_MR, "Pred")],
                "sequence": value["seq"],
                "split": split,
            }

            TKV_data[patient_MR] = summary

    df = pd.DataFrame(TKV_data).transpose()

    if output is not None:
        df.to_csv(output)

    return TKV_data


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="YAML config path", type=str, required=True
    )
    parser.add_argument(
        "--makelinks", help="Make data links", action="store_true"
    )
    parser.add_argument(
        "--out_path", help="Path to output csv", required=True)

    args = parser.parse_args()
    calculate_TKVs(args.run_makelinks, args.out_path, args.config)
