"""Check prediction entropy and dice score correlation"""

# %%
import os
import yaml

import matplotlib.pyplot as plt
import seaborn as sns

import torch

# enable lib loading even if not installed as a pip package or in PYTHONPATH
# also convenient for relative paths in example config files
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

from adpkd_segmentation.config.config_utils import get_object_instance # noqa
from adpkd_segmentation.evaluate import validate # noqa
from adpkd_segmentation.utils.train_utils import load_model_data # noqa

sns.set()

# %%
CONFIG = "experiments/september02/random_split_new_data_less_albu_10_more/val/val.yaml" # noqa

with open(CONFIG, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# %%
model_config = config["_MODEL_CONFIG"]
dataloader_config = config["_VAL_DATALOADER_CONFIG"]
losses_config = config["_LOSSES_METRICS_CONFIG"]
saved_checkpoint = config["_MODEL_CHECKPOINT"]
# override
dataloader_config["batchsize"] = 1

# %%
model = get_object_instance(model_config)()
load_model_data(saved_checkpoint, model, True)
dataloader = get_object_instance(dataloader_config)()
loss_metric = get_object_instance(losses_config)()

# %%
device = torch.device("cuda:0")
model = model.to(device)
model.eval()

# %%
averaged, all_losses_and_metrics = validate(
    dataloader, model, loss_metric, device, output_losses_list=True)

# %%
dice_scores = all_losses_and_metrics["dice_metric"]
entropy = all_losses_and_metrics["prediction_entropy"]

plt.xlabel("Prediction entropy")
plt.ylabel("Dice score")
sns.scatterplot(x=entropy, y=dice_scores)

# %%
