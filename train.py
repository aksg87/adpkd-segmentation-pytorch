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
import torch.optim as optim

from config.config_utils import get_object_instance
from data.link_data import makelinks
from evaluate import validate


# %%
def get_losses_str(losses_and_metrics, tensors=True):
    out = []
    for k, v in losses_and_metrics.items():
        if tensors:
            v = v.item()
        out.append("{}:{:.5f}".format(k, v))
    return ", ".join(out)


def is_better(current, previous, metric_type):
    if metric_type == "low":
        return current < previous
    elif metric_type == "high":
        return current > previous
    else:
        raise Exception("unknown metric_type")


# %%
def train(config):
    model_config = config["_MODEL_CONFIG"]
    dataloader_config = config["_DATALOADER_CONFIG"]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]
    results_path = config["_RESULTS_PATH"]
    saved_checkpoint = config["_MODEL_CHECKPOINT"]
    loss_key = config["_OPTIMIZATION_LOSS"]
    lr_scheduler_config = config["_LR_SCHEDULER"]

    model = get_object_instance(model_config)()
    model.load_state_dict(torch.load(saved_checkpoint))

    dataloaders = get_object_instance(dataloader_config)()
    loss_metric = get_object_instance(loss_metric_config)()
    lr_scheduler_getter = get_object_instance(lr_scheduler_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO move to config
    optim_param_dict = {"lr": 0.001}
    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad, model.parameters(), **optim_param_dict
        )
    )
    lr_scheduler = lr_scheduler_getter(optimizer)

    # train, val, test ordering
    train_loader = dataloaders[0]
    val_loader = dataloaders[1]

    model = model.to(device)
    model.train()
    # TODO config
    num_epochs = 2
    batch_log_interval = 100
    best_metric_type = "low"  # "high" or "low"
    saving_metric = "baseline_loss"
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
            if (idx + 1) % batch_log_interval == 0:
                print(get_losses_str(losses_and_metrics))

        # done with one epoch
        # let's validate (use code from the validation script)
        model.eval()
        all_losses_and_metrics = validate(
            val_loader, model, loss_metric, device
        )
        print(get_losses_str(all_losses_and_metrics))
        model.train()
        current = all_losses_and_metrics[saving_metric]
        if is_better(current, previous, best_metric_type):
            print("Validation metric improved"
                  "at the end of epoch {}".format(epoch))
            previous = current
            # TODO
            # save the model
            # store validation stats in json

        # learning rate schedule step at the end of epoch
        # TODO: generalize to support LR schedulers which use
        # `.step(epoch)``
        if config["_LR_SCHEDULER_USE_VAL"]:
            lr_scheduler.step(all_losses_and_metrics[loss_key])
        else:
            lr_scheduler.step()


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

    # %%
    makelinks()
    train(config)
