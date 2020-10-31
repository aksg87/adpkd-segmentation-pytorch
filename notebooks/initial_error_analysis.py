"""Check low dice examples"""

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

from adpkd_segmentation.config.config_utils import get_object_instance  # noqa
from adpkd_segmentation.evaluate import validate  # noqa
from adpkd_segmentation.utils.train_utils import load_model_data  # noqa

sns.set()

# %%
# CONFIG = "experiments/september/11_stratified_albu_v2_b4_simple_norm/val/val.yaml"  # noqa
# CONFIG = "./experiments/september06/random_split_new_data_less_albu/test/test.yaml"
CONFIG = "./experiments/september03/random_split_new_data_less_albu/test/test.yaml"
with open(CONFIG, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# %%
model_config = config["_MODEL_CONFIG"]
dataloader_config = config["_TEST_DATALOADER_CONFIG"]
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
device = torch.device("cpu:0")
model = model.to(device)
model.eval()

# %%
averaged, all_losses_and_metrics = validate(
    dataloader, model, loss_metric, device, output_losses_list=True
)

# %%
dice_scores = all_losses_and_metrics["dice_metric"]

# %%
plt.xlabel("Example index")
plt.ylabel("Dice score")
sns.scatterplot(x=range(len(dice_scores)), y=dice_scores, alpha=0.4)

# %%
very_low_dice = [
    (dice, idx) for idx, dice in enumerate(dice_scores) if dice < 0.05
]

# %%
# check one example
image, mask, idx = dataloader.dataset[very_low_dice[1][1]]
print(mask.sum())
im_tensor = torch.from_numpy(image).unsqueeze(0)
# %%
pred = model(im_tensor)  # (1, 1, 224, 224)
# %%
im = image[0]  # (3, 224, 224) original
msk = mask[0]  # (1, 224, 224) original
pred_sigm = torch.sigmoid(pred)[0][0].detach().numpy()

# %%
plt.rcParams["axes.grid"] = False
f, axarr = plt.subplots(1, 3)
axarr[0].imshow(im, cmap="gray")
axarr[1].imshow(im, cmap="gray")  # background for mask
axarr[1].imshow(msk, alpha=0.7)
axarr[2].imshow(im, cmap="gray")  # background for mask
axarr[2].imshow(pred_sigm, alpha=0.7)

# %%
# check all low dice
for dice, idx in very_low_dice:
    image, mask, _ = dataloader.dataset[idx]
    im_tensor = torch.from_numpy(image).unsqueeze(0)
    pred = model(im_tensor)
    im = image[0]  # (3, 224, 224) original
    msk = mask[0]  # (1, 224, 224) original
    pred_sigm = torch.sigmoid(pred)[0][0].detach().numpy()

    plt.rcParams["axes.grid"] = False
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(im, cmap="gray")
    axarr[1].imshow(im, cmap="gray")  # background for mask
    axarr[1].imshow(msk, alpha=0.5)
    axarr[2].imshow(im, cmap="gray")  # background for mask
    axarr[2].imshow(pred_sigm, alpha=0.5)

# %%
middle_dice = [
    (dice, idx)
    for idx, dice in enumerate(dice_scores)
    if dice > 0.05 and dice < 0.8
]


# %%
def check_prediction(idx):
    image, mask, _ = dataloader.dataset[idx]
    im_tensor = torch.from_numpy(image).unsqueeze(0)
    pred = model(im_tensor)
    im = image[0]  # (3, 224, 224) original
    msk = mask[0]  # (1, 224, 224) original
    pred_sigm = torch.sigmoid(pred)[0][0].detach().numpy()

    plt.rcParams["axes.grid"] = False
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(im, cmap="gray")
    axarr[1].imshow(im, cmap="gray")  # background for mask
    axarr[1].imshow(msk, alpha=0.5)
    axarr[2].imshow(im, cmap="gray")  # background for mask
    axarr[2].imshow(pred_sigm, alpha=0.5)


# %%
for dice, idx in middle_dice:
    check_prediction(idx)


# %%
def get_patients(dice_scores, dataset):
    patients = []
    for score, idx in dice_scores:
        _, _, attribs = dataset.get_verbose(idx)
        patients.append(attribs["patient"])
    return patients

# %%
