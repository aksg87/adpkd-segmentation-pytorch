"""
Model evaluation script

python -m evaluate --config path_to_config_yaml
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


# %%
def validate(dataloader, model, loss_metric, device):
    all_losses_and_metrics = defaultdict(list)
    num_examples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_batch_hat = model(x_batch)

        batch_size = y_batch.size(0)
        num_examples += batch_size
        losses_and_metrics = loss_metric(y_batch_hat, y_batch)
        for key, value in losses_and_metrics.items():
            all_losses_and_metrics[key].append(value.item() * batch_size)

    for key, value in all_losses_and_metrics.items():
        all_losses_and_metrics[key] = (
            sum(all_losses_and_metrics[key]) / num_examples
        )
    return all_losses_and_metrics


# %%
def evaluate(config):
    model_config = config["_MODEL_CONFIG"]
    dataloader_config = config["_DATALOADER_CONFIG"]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]
    results_path = config["_RESULTS_PATH"]
    saved_checkpoint = config["_MODEL_CHECKPOINT"]

    model = get_object_instance(model_config)()
    model.load_state_dict(torch.load(saved_checkpoint))

    dataloaders = get_object_instance(dataloader_config)()
    loss_metric = get_object_instance(loss_metric_config)()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: add as a config option
    # train, val, test ordering
    dataloader_index = 1
    dataloader = dataloaders[dataloader_index]

    model = model.to(device)
    model.eval()
    all_losses_and_metrics = validate(dataloader, model, loss_metric, device)

    os.makedirs(results_path, exist_ok=True)
    with open("{}/val_results.json".format(results_path), "w") as fp:
        print(all_losses_and_metrics)
        json.dump(all_losses_and_metrics, fp, indent=4)


# %%
def quick_check(run_makelinks=False):
    if run_makelinks:
        makelinks()
    path = "./config/examples/eval_example.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    evaluate(config)


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
    evaluate(config)
