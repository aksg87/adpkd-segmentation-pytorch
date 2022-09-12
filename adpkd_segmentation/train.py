"""
Model train script

python -m adpkd_segmentation.train --config --config path_to_config_yaml --makelinks # noqa

If using a specific GPU (e.g. device 2):
CUDA_VISIBLE_DEVICES=2 python -m train --config path_to_config_yaml --makelinks

The makelinks flag is needed only once to create symbolic links to the data.

To create and activate the conda environment for this repository:
conda env create --file adpkd-segmentation.yml
conda activate adpkd-segmentation
"""

# %%
import argparse
import json
import numpy as np
import os
import yaml
import random
from collections import OrderedDict

from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from catalyst.contrib.nn import Lookahead

from adpkd_segmentation.create_eval_configs import create_config
from adpkd_segmentation.config.config_utils import get_object_instance
from adpkd_segmentation.data.link_data import makelinks
from adpkd_segmentation.evaluate import validate
from adpkd_segmentation.utils.train_utils import (
    load_model_data,
    save_model_data,
)
from adpkd_segmentation.data.data_utils import tensor_dict_to_device

CHECKPOINTS = "checkpoints"
RESULTS = "results"
TB_LOGS = "tb_logs"


# %%
def get_losses_str(losses_and_metrics, tensors=True):
    out = []
    for k, v in losses_and_metrics.items():
        if tensors:
            v = v.item()
        out.append("{}:{:.5f}".format(k, v))
    return ", ".join(out)


# %%
def is_better(current, previous, metric_type):
    if metric_type == "low":
        return current < previous
    elif metric_type == "high":
        return current > previous
    else:
        raise Exception("unknown metric_type")


# %%
def save_val_metrics(metrics, results_dir, epoch, global_step):
    with open(
        "{}/val_results_ep_{}_gs_{}.json".format(
            results_dir, epoch, global_step
        ),
        "w",
    ) as fp:
        json.dump(metrics, fp, indent=4)


# %%
def tb_log_metrics(writer, metrics, global_step):
    for k, v in metrics.items():
        writer.add_scalar(k, v, global_step)


# %%
def plot_image_from_batch(
    writer, batch, prediction, target, global_step, idx=0
):
    image = batch[idx]
    pred_mask = prediction[idx]
    mask = target[idx]
    # plot each of the image channels as a separate grayscale image
    # (C, 1, H, W) shape to treat each of the channels as a new image
    image = image.unsqueeze(1)
    # same approach for mask and prediction
    mask = mask.unsqueeze(1)
    pred_mask = pred_mask.unsqueeze(1)
    writer.add_images("input_images", image, global_step, dataformats="NCHW")
    writer.add_images("mask_channels", mask, global_step, dataformats="NCHW")
    writer.add_images(
        "prediction_channels", pred_mask, global_step, dataformats="NCHW"
    )


# %%
def plot_fig_from_batch(
    writer,
    batch,
    prediction,
    target,
    global_step,
    idx=0,
    title="fig: img_target_pred",
):
    image = batch[idx][1]  # middle channel
    # single channel by default
    mask = target[idx][0]
    pred_mask = prediction[idx][0]
    # TODO add standardizaton to convert all setups into single channel format
    # currently supporting 1 and 2 channels, sigmoid based
    num_channels = target.shape[1]
    if num_channels == 2:
        pred_mask = prediction[idx][0] + prediction[idx][1]  # combine channels
        mask = target[idx][0] + target[idx][1]  # combine channels

    image = image.detach().cpu()
    mask = mask.detach().cpu()
    pred_mask = pred_mask.detach().cpu()

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(image, cmap="gray")
    axarr[1].imshow(image, cmap="gray")  # background for mask
    axarr[1].imshow(mask, alpha=0.5)
    axarr[2].imshow(image, cmap="gray")  # background for mask
    axarr[2].imshow(pred_mask, alpha=0.5)

    writer.add_figure(title, f, global_step)


# %%
def train(config, config_save_name):
    # reproducibility
    seed = config.get("_SEED", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_config = config["_MODEL_CONFIG"]
    train_dataloader_config = config["_TRAIN_DATALOADER_CONFIG"]
    val_dataloader_config = config["_VAL_DATALOADER_CONFIG"]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]
    experiment_dir = config["_EXPERIMENT_DIR"]
    checkpoints_dir = os.path.join(experiment_dir, CHECKPOINTS)
    results_dir = os.path.join(experiment_dir, RESULTS)
    tb_logs_dir_train = os.path.join(experiment_dir, TB_LOGS, "train")
    tb_logs_dir_val = os.path.join(experiment_dir, TB_LOGS, "val")
    config_out = os.path.join(experiment_dir, config_save_name)

    saved_checkpoint = config["_MODEL_CHECKPOINT"]
    checkpoint_format = config["_NEW_CKP_FORMAT"]
    loss_key = config["_OPTIMIZATION_LOSS"]
    optim_config = config["_OPTIMIZER"]
    lookahead_config = config["_LOOKAHEAD_OPTIM"]
    lr_scheduler_config = config["_LR_SCHEDULER"]
    experiment_data = config["_EXPERIMENT_DATA"]
    val_plotting_dict = config.get("_VAL_PLOTTING")

    model = get_object_instance(model_config)()
    global_step = 0
    if saved_checkpoint is not None:
        global_step = load_model_data(
            saved_checkpoint, model, new_format=checkpoint_format
        )
    train_loader = get_object_instance(train_dataloader_config)()
    val_loader = get_object_instance(val_dataloader_config)()

    print("Train dataset length: {}".format(len(train_loader.dataset)))
    print("Validation dataset length: {}".format(len(val_loader.dataset)))
    print(
        "Valiation dataset patients:\n{}".format(val_loader.dataset.patients)
    )

    loss_metric = get_object_instance(loss_metric_config)()
    optimizer_getter = get_object_instance(optim_config)
    lr_scheduler_getter = get_object_instance(lr_scheduler_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoints_dir)
    os.makedirs(results_dir)
    os.makedirs(tb_logs_dir_train)
    os.makedirs(tb_logs_dir_val)
    with open(config_out, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # create configs for val and test
    val_config, val_out_dir = create_config(config, "val")
    test_config, test_out_dir = create_config(config, "test")
    os.makedirs(val_out_dir)
    os.makedirs(test_out_dir)

    val_path = os.path.join(val_out_dir, "val.yaml")
    print("Creating evaluation config for val: {}".format(val_path))
    with open(val_path, "w") as f:
        yaml.dump(val_config, f, default_flow_style=False)

    test_path = os.path.join(test_out_dir, "test.yaml")
    print("Creating evaluation config for test: {}".format(test_path))
    with open(test_path, "w") as f:
        yaml.dump(test_config, f, default_flow_style=False)

    train_writer = SummaryWriter(tb_logs_dir_train)
    val_writer = SummaryWriter(tb_logs_dir_val)

    model_params = model.parameters()
    if config.get("_MODEL_PARAM_PREP") is not None:
        model_prep = get_object_instance(config.get("_MODEL_PARAM_PREP"))
        model_params = model_prep(model)

    optimizer = optimizer_getter(model_params)
    if lookahead_config["use_lookahead"]:
        optimizer = Lookahead(optimizer, **lookahead_config["params"])
    lr_scheduler = lr_scheduler_getter(optimizer)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    model.train()
    num_epochs = experiment_data["num_epochs"]
    batch_log_interval = experiment_data["batch_log_interval"]
    # "low" or "high"
    best_metric_type = experiment_data["best_metric_type"]
    saving_metric = experiment_data["saving_metric"]
    previous = float("inf") if best_metric_type == "low" else float("-inf")

    output_example_idx = (
        hasattr(train_loader.dataset, "output_idx")
        and train_loader.dataset.output_idx
    )

    for epoch in range(num_epochs):
        for output in train_loader:
            if output_example_idx:
                x_batch, y_batch, index = output
                extra_dict = train_loader.dataset.get_extra_dict(index)
                extra_dict = tensor_dict_to_device(extra_dict, device)
            else:
                x_batch, y_batch = output
                extra_dict = None

            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch_hat = model(x_batch)
            losses_and_metrics = loss_metric(y_batch_hat, y_batch, extra_dict)
            loss = losses_and_metrics[loss_key]
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % batch_log_interval == 0:
                print("TRAIN:", get_losses_str(losses_and_metrics))
                tb_log_metrics(train_writer, losses_and_metrics, global_step)
                # TODO: add support for softmax processing
                prediction = torch.sigmoid(y_batch_hat)
                plot_fig_from_batch(
                    train_writer, x_batch, prediction, y_batch, global_step,
                )
            # lr change after each batch
            if lr_scheduler_getter.step_type == "after_batch":
                lr_scheduler.step()

        # done with one epoch
        # let's validate (use code from the validation script)
        model.eval()
        all_losses_and_metrics = validate(
            val_loader,
            model,
            loss_metric,
            device,
            plotting_func=plot_fig_from_batch,
            plotting_dict=val_plotting_dict,
            writer=val_writer,
            global_step=global_step,
            val_metric_to_check=saving_metric,
            output_losses_list=False,
        )

        print("Validation results for epoch {}".format(epoch))
        print("VAL:", get_losses_str(all_losses_and_metrics, tensors=False))
        model.train()

        current = all_losses_and_metrics[saving_metric]
        if is_better(current, previous, best_metric_type):
            print(
                "Validation metric improved "
                "at the end of epoch {}".format(epoch)
            )
            previous = current
            save_val_metrics(
                all_losses_and_metrics, results_dir, epoch, global_step
            )
            out_path = os.path.join(checkpoints_dir, "best_val_checkpoint.pth")
            save_model_data(out_path, model, global_step)

        tb_log_metrics(val_writer, all_losses_and_metrics, global_step)

        # learning rate schedule step at the end of epoch
        if lr_scheduler_getter.step_type != "after_batch":
            if lr_scheduler_getter.step_type == "use_val":
                lr_scheduler.step(all_losses_and_metrics[loss_key])
            elif lr_scheduler_getter.step_type == "use_epoch":
                lr_scheduler.step(epoch)
            else:
                lr_scheduler.step()

        # plot distinct learning rates in order they appear in the optimizer
        lr_dict = OrderedDict()
        for param_group in optimizer.param_groups:
            lr = param_group.get("lr")
            lr_dict[lr] = None
        for idx, lr in enumerate(lr_dict):
            tb_log_metrics(val_writer, {"lr_{}".format(idx): lr}, global_step)
            tb_log_metrics(
                train_writer, {"lr_{}".format(idx): lr}, global_step
            )

    train_writer.close()
    val_writer.close()


# %%
def quick_check(run_makelinks=False):
    if run_makelinks:
        makelinks()
    path = "./config/examples/train_example.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    train(config)


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
    config_save_name = os.path.basename(args.config)
    train(config, config_save_name)
