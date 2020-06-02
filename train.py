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

CHECKPOINTS = "checkpoints"
RESULTS = "results"
TF_LOGS = "tf_logs"


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
def tf_log_metrics(writer, metrics, global_step, suffix):
    for k, v in metrics.items():
        k = k + "/" + suffix
        writer.add_scalar(k, v, global_step)


# %%
def train(config):
    model_config = config["_MODEL_CONFIG"]
    dataloader_config = config["_DATALOADER_CONFIG"]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]
    experiment_dir = config["_EXPERIMENT_DIR"]
    checkpoints_dir = os.path.join(experiment_dir, CHECKPOINTS)
    results_dir = os.path.join(experiment_dir, RESULTS)
    tf_logs_dir = os.path.join(experiment_dir, TF_LOGS)

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
    dataloaders = get_object_instance(dataloader_config)()
    loss_metric = get_object_instance(loss_metric_config)()
    optimizer_getter = get_object_instance(optim_config)
    lr_scheduler_getter = get_object_instance(lr_scheduler_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tf_logs_dir, exist_ok=True)

    writer = SummaryWriter(tf_logs_dir)

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optimizer_getter(model_params)
    if lookahead_config["use_lookahead"]:
        optimizer = Lookahead(optimizer, **lookahead_config["params"])
    lr_scheduler = lr_scheduler_getter(optimizer)

    # train, val, test ordering
    train_loader = dataloaders[0]
    val_loader = dataloaders[1]

    model = model.to(device)
    model.train()
    num_epochs = experiment_data["num_epochs"]
    batch_log_interval = experiment_data["batch_log_interval"]
    # "low" or "high"
    best_metric_type = experiment_data["best_metric_type"]
    saving_metric = "loss_baseline"
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
                print(get_losses_str(losses_and_metrics))
                tf_log_metrics(
                    writer, losses_and_metrics, global_step, "train"
                )

        # done with one epoch
        # let's validate (use code from the validation script)
        model.eval()
        all_losses_and_metrics = validate(
            val_loader, model, loss_metric, device
        )
        print("Validation results for epoch {}".format(epoch))
        print(get_losses_str(all_losses_and_metrics, tensors=False))
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
            out_path = os.path.join(
                checkpoints_dir, "ckp_gs_{}.pth".format(global_step)
            )
            save_model_data(out_path, model, global_step)
        tf_log_metrics(writer, all_losses_and_metrics, global_step, "val")

        # learning rate schedule step at the end of epoch
        if lr_scheduler_getter.step_type == "use_val":
            lr_scheduler.step(all_losses_and_metrics[loss_key])
        elif lr_scheduler_getter.step_type == "use_epoch":
            lr_scheduler.step(epoch)
        else:
            lr_scheduler.step()

    writer.close()


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

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    makelinks()
    train(config)
