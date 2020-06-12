"""
Model train script

python -m train --config path_to_config_yaml
"""

# %%
import argparse
import json
import os
import yaml

import torch
from torch.utils.tensorboard import SummaryWriter
from catalyst.contrib.nn import Lookahead

from config.config_utils import get_object_instance
from data.link_data import makelinks
from evaluate import validate
from train_utils import load_model_data, save_model_data
from matplotlib import pyplot as plt

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


def plot_fig_from_batch(writer, batch, prediction, target, global_step, idx=0):
    image = batch[idx][1]  # middle channel
    pred_mask = prediction[idx][0] + prediction[idx][1]  # combine channels
    mask = target[idx][0] + target[idx][1]  # combine channels

    # print(f"***shapes {image.shape} {pred_mask.shape} {mask.shape}")
    # plot each of the image channels as a separate grayscale image
    # (C, 1, H, W) shape to treat each of the channels as a new image
    image = image.cpu()
    mask = mask.cpu()
    pred_mask = pred_mask.cpu().detach()

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(image, cmap="gray")
    axarr[1].imshow(image, cmap="gray")  # background for mask
    axarr[1].imshow(mask, alpha=0.5)
    axarr[2].imshow(image, cmap="gray")  # background for mask
    axarr[2].imshow(pred_mask, alpha=0.5)

    writer.add_figure("fig: img_target_pred", f, global_step)


# %%
def train(config):
    model_config = config["_MODEL_CONFIG"]
    train_dataloader_config = config["_TRAIN_DATALOADER_CONFIG"]
    val_dataloader_config = config["_VAL_DATALOADER_CONFIG"]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]
    experiment_dir = config["_EXPERIMENT_DIR"]
    checkpoints_dir = os.path.join(experiment_dir, CHECKPOINTS)
    results_dir = os.path.join(experiment_dir, RESULTS)
    tb_logs_dir_train = os.path.join(experiment_dir, TB_LOGS, "train")
    tb_logs_dir_val = os.path.join(experiment_dir, TB_LOGS, "val")

    saved_checkpoint = config["_MODEL_CHECKPOINT"]
    checkpoint_format = config["_NEW_CKP_FORMAT"]
    loss_key = config["_OPTIMIZATION_LOSS"]
    optim_config = config["_OPTIMIZER"]
    lookahead_config = config["_LOOKAHEAD_OPTIM"]
    lr_scheduler_config = config["_LR_SCHEDULER"]
    experiment_data = config["_EXPERIMENT_DATA"]

    model = get_object_instance(model_config)()
    global_step = 0
    if saved_checkpoint is not None:
        global_step = load_model_data(
            saved_checkpoint, model, new_format=checkpoint_format
        )
    train_loader = get_object_instance(train_dataloader_config)()
    val_loader = get_object_instance(val_dataloader_config)()
    loss_metric = get_object_instance(loss_metric_config)()
    optimizer_getter = get_object_instance(optim_config)
    lr_scheduler_getter = get_object_instance(lr_scheduler_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tb_logs_dir_train, exist_ok=True)
    os.makedirs(tb_logs_dir_val, exist_ok=True)

    train_writer = SummaryWriter(tb_logs_dir_train)
    val_writer = SummaryWriter(tb_logs_dir_val)

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer_getter(model_params)
    if lookahead_config["use_lookahead"]:
        optimizer = Lookahead(optimizer, **lookahead_config["params"])
    lr_scheduler = lr_scheduler_getter(optimizer)

    model = model.to(device)
    model.train()
    num_epochs = experiment_data["num_epochs"]
    batch_log_interval = experiment_data["batch_log_interval"]
    # "low" or "high"
    best_metric_type = experiment_data["best_metric_type"]
    saving_metric = experiment_data["saving_metric"]
    previous = float("inf") if best_metric_type == "low" else float("-inf")

    for epoch in range(num_epochs):
        for idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch_hat = model(x_batch)
            losses_and_metrics = loss_metric(y_batch_hat, y_batch)
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
                    train_writer, x_batch, prediction, y_batch, global_step
                )

        # done with one epoch
        # let's validate (use code from the validation script)
        model.eval()
        all_losses_and_metrics = validate(
            val_loader, model, loss_metric, device
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
        if lr_scheduler_getter.step_type == "use_val":
            lr_scheduler.step(all_losses_and_metrics[loss_key])
        elif lr_scheduler_getter.step_type == "use_epoch":
            lr_scheduler.step(epoch)
        else:
            lr_scheduler.step()
        # plot the learning rate
        for idx, param_group in enumerate(optimizer.param_groups):
            key = "lr_param_group_{}".format(idx)
            value = param_group["lr"]
            tb_log_metrics(val_writer, {key: value}, global_step)
            tb_log_metrics(train_writer, {key: value}, global_step)

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
    train(config)
