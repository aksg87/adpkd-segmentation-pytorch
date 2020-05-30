"""
Model train script

python -m train --config path_to_config_yaml
"""

# %%
from collections import defaultdict
import argparse
import yaml
import pickle

import torch
import torch.optim as optim

from config.config_utils import get_object_instance
from data.link_data import makelinks

# %%
makelinks()

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

    # TODO: add as a config option
    # train, val, test ordering
    dataloader_index = 1
    dataloader = dataloaders[dataloader_index]

    model = model.to(device)
    model.train()
    all_losses_and_metrics = defaultdict(list)

    num_examples = 0
    num_epochs = 30

    for epoch in range(num_epochs):
        for x_batch, y_batch in dataloader:

            model.train()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch_hat = model(x_batch)

            batch_size = y_batch.size(0)
            num_examples += batch_size
            losses_and_metrics = loss_metric(y_batch_hat, y_batch)

            for key, value in losses_and_metrics.items():
                all_losses_and_metrics[key].append(value * batch_size)

            loss = torch.mean(
                torch.stack(all_losses_and_metrics[loss_key])
            )  # TODO: Double check this
            loss.backward()  # FIXME: RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed.
            optimizer.step()
            optimizer.zero_grad()

    for key, value in all_losses_and_metrics.items():
        all_losses_and_metrics[key] = (
            torch.sum(torch.stack(all_losses_and_metrics[key]))
            / num_examples  # TODO, use torch.mean?
        )

    with open("{}/train_results.pickle".format(results_path), "wb") as fp:
        print("{}/train_results.pickle")
        pickle.dump(all_losses_and_metrics, fp)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="YAML config path", type=str, required=True
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train(config)

# uncomment and run for a quick check
# %%
path = "./config/examples/eval_example.yaml"
with open(path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
train(config)

# %%
