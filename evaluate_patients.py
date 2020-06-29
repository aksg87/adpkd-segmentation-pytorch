"""
Model evaluation script

python -m evaluate --config path_to_config_yaml --makelinks

If using a specific GPU (e.g. device 2):
CUDA_VISIBLE_DEVICES=2 python -m evaluate --config path_to_config_yaml

The makelinks flag is needed only once to create symbolic links to the data.
"""

# %%
from collections import defaultdict
import argparse
import json
import os
import yaml

import torch

from config.config_utils import get_object_instance
from data.link_data import makelinks
from data.data_utils import masks_to_colorimg
from matplotlib import pyplot as plt
from train_utils import load_model_data


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
):
    all_losses_and_metrics = defaultdict(list)
    num_examples = 0

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        batch_size = y_batch.size(0)
        num_examples += batch_size
        with torch.no_grad():
            y_batch_hat = model(x_batch)
            losses_and_metrics = loss_metric(y_batch_hat, y_batch)

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

    averaged = {}
    for key, value in all_losses_and_metrics.items():
        averaged[key] = sum(all_losses_and_metrics[key]) / num_examples

    if output_losses_list:
        return averaged, all_losses_and_metrics
    return averaged


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    all_losses_and_metrics = validate(dataloader, model, loss_metric, device)

    os.makedirs(results_path)
    with open("{}/val_results.json".format(results_path), "w") as fp:
        print(all_losses_and_metrics)
        json.dump(all_losses_and_metrics, fp, indent=4)

    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    inputs = inputs.to(device)
    preds = model(inputs)
    inputs = inputs.cpu()
    preds = preds.cpu()

    plot_figure_from_batch(inputs, preds)


# %%
def plot_figure_from_batch(inputs, preds, target=None, idx=0):

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(inputs[idx][1], cmap="gray")
    axarr[1].imshow(inputs[idx][1], cmap="gray")  # background for mask
    axarr[1].imshow(masks_to_colorimg(preds[idx]), alpha=0.5)

    return f


# %%
def quick_check(run_makelinks=False):
    if run_makelinks:
        makelinks()
    path = "./experiments/june28/train_example_all_no_noise_patient_seq_norm_b5_BN/test/test.yaml"

    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    evaluate(config)


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
