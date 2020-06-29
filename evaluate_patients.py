"""
Model evaluation script

python -m evaluate --config path_to_config_yaml --makelinks

If using a specific GPU (e.g. device 2):
CUDA_VISIBLE_DEVICES=2 python -m evaluate --config path_to_config_yaml

The makelinks flag is needed only once to create symbolic links to the data.
"""

# %%
from collections import OrderedDict, defaultdict
import argparse
import json
import os
import yaml
import pandas as pd

import torch

from config.config_utils import get_object_instance
from data.link_data import makelinks
from data.data_utils import masks_to_colorimg
from matplotlib import pyplot as plt
from train_utils import load_model_data
from new_datasets import dataloader


# %%
def validate(
    dataloader,
    model,
    loss_metric,
    device,
    plotting_func=None,
    plotting_dict=None,
    writer=None,
    global_step=None,
    val_metric_to_check=None,
    output_losses_list=False,
    binarize_func=None,
):
    all_losses_and_metrics = defaultdict(list)
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
            losses_and_metrics = loss_metric(y_batch_hat, y_batch)

            start_idx = num_examples - batch_size
            end_idx = num_examples

            for inbatch_idx, dataset_idx in enumerate(
                range(start_idx, end_idx)
            ):
                # calculate TKV and TKV inputs for each dcm
                _, dcm_path, attribs = dataset.get_verbose(dataset_idx)
                attribs["pred_kidney_pixels"] = torch.sum(
                    y_batch_hat_binary[inbatch_idx] > 0
                ).item()

                # TODO: Clean up method of accessing Resize transform
                attribs["transform_resize_dim"] = (
                    dataloader.dataset.augmentation[0].height,
                    dataloader.dataset.augmentation[0].width,
                )

                # scale factor takes differences in pred^2 / GT^2 based on img dim or pixel area
                scale_factor = (attribs["dim"][0] ** 2) / (
                    attribs["transform_resize_dim"][0] ** 2
                )
                attribs["Vol_GT"] = (
                    attribs["vox_vol"] * attribs["kidney_pixels"]
                )
                attribs["Vol_Pred"] = (
                    scale_factor
                    * attribs["vox_vol"]
                    * attribs["pred_kidney_pixels"]
                )

                updated_dcm2attribs[dcm_path] = attribs

            for key, value in losses_and_metrics.items():
                all_losses_and_metrics[key].append(value.item() * batch_size)

            if plotting_dict is not None and batch_idx in plotting_dict:
                # TODO: add support for softmax processing
                prediction = torch.sigmoid(y_batch_hat)
                image_idx = plotting_dict[batch_idx]
                plotting_func(
                    writer=writer,
                    batch=x_batch,
                    prediction=prediction,
                    target=y_batch,
                    global_step=global_step,
                    idx=image_idx,
                    title="val_batch_{}_image_{}".format(batch_idx, image_idx),
                )
                # check DSC metric for this image
                # `loss_metric` expects raw model outputs without the sigmoid
                im_pred = y_batch_hat[image_idx].unsqueeze(0)
                im_target_mask = y_batch[image_idx].unsqueeze(0)
                im_losses = loss_metric(im_pred, im_target_mask)
                writer.add_scalar(
                    "val_batch_{}_image_{}_{}".format(
                        batch_idx, image_idx, val_metric_to_check
                    ),
                    im_losses[val_metric_to_check],
                    global_step,
                )

    return updated_dcm2attribs


# %%
def evaluate(config):
    model_config = config["_MODEL_CONFIG"]
    loader_to_eval = config["_LOADER_TO_EVAL"]
    dataloader_config = config[loader_to_eval]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]
    results_path = config["_RESULTS_PATH"]
    saved_checkpoint = config["_MODEL_CHECKPOINT"]
    checkpoint_format = config["_NEW_CKP_FORMAT"]

    model = get_object_instance(model_config)()
    if saved_checkpoint is not None:
        load_model_data(saved_checkpoint, model, new_format=checkpoint_format)

    dataloader = get_object_instance(dataloader_config)()

    loss_metric = get_object_instance(loss_metric_config)()

    criterions_dict = get_object_instance(loss_metric_config).criterions_dict

    # TODO: Hardcoded to dice_metric, so support other metrics from config.
    binarize_func = criterions_dict["dice_metric"].binarize_func

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    updated_dcm2attribs = validate(
        dataloader, model, loss_metric, device, binarize_func=binarize_func
    )

    return updated_dcm2attribs


# %%
def plot_figure_from_batch(inputs, preds, target=None, idx=0):

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(inputs[idx][1], cmap="gray")
    axarr[1].imshow(inputs[idx][1], cmap="gray")  # background for mask
    axarr[1].imshow(masks_to_colorimg(preds[idx]), alpha=0.5)

    return f


# %%
def calculate_TKVs(run_makelinks=False, output=None):
    if run_makelinks:
        makelinks()
    path = "./experiments/june28/train_example_all_no_noise_patient_seq_norm_b5_BN/test/test.yaml"

    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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
                # TODO: automatically determine val/test
                "split": "test",
            }

            TKV_data[patient_MR] = summary

    df = pd.DataFrame(TKV_data).transpose()

    if not output is None:
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

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.makelinks:
        makelinks()

    evaluate(config)
