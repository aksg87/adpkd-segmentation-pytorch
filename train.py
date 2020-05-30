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


# %%
def train(config):
    model_config = config["_MODEL_CONFIG"]
    dataloader_config = config["_DATALOADER_CONFIG"]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]
    results_path = config["_RESULTS_PATH"]
    saved_checkpoint = config["_MODEL_CHECKPOINT"]

    loss_key = "loss_baseline"  # TODO move to config

    model = get_object_instance(model_config)()
    model.load_state_dict(torch.load(saved_checkpoint))

    dataloaders = get_object_instance(dataloader_config)()
    loss_metric = get_object_instance(loss_metric_config)()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters())
    )  # TODO move to config

    # train, val, test ordering
    train_loader = dataloaders[0]
    val_loader = dataloaders[1]

    model = model.to(device)
    model.train()
    num_epochs = 2  # TODO config

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch_hat = model(x_batch)
            losses_and_metrics = loss_metric(y_batch_hat, y_batch)
            loss = losses_and_metrics[loss_key]
            loss.backward()
            optimizer.step()
            # print(loss.item())
            # we want to print all in `losses_and_metrics`
            # let's add batch indexing, so for e.g. every 1000th batch

        # done with one epoch
        # let's validate (use code from the validation script)

        # save the model if best by some validation metric
        # save best validation stats in json

        # learning rate schedule step at the end of epoch




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

# uncomment and run for a quick check
# %%
# makelinks()
# path = "./config/examples/eval_example.yaml"
# with open(path, "r") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)

# train(config)
