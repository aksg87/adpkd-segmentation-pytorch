"""Basic checks for the evaluate.py script"""

# %%
import matplotlib.pyplot as plt
import yaml
import os

# enable lib loading even if not installed as a pip package or in PYTHONPATH
# also convenient for relative paths in example config files
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)

from adpkd_segmentation.config.config_utils import get_object_instance  # noqa
from adpkd_segmentation.data.link_data import makelinks  # noqa
from adpkd_segmentation.data.data_utils import ( # noqa
    masks_to_colorimg,
    display_traindata,
)

# %% needed only once
# makelinks()


# %%
def evaluate(config):
    model_config = config["_MODEL_CONFIG"]
    dataloader_config = config["_VAL_DATALOADER_CONFIG"]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]
    model = get_object_instance(model_config)()
    dataloader = get_object_instance(dataloader_config)()
    loss_metric = get_object_instance(loss_metric_config)()

    return (
        model,
        dataloader,
        loss_metric,
    )  # add return types for debugging/testing


# %%
path = "./misc/example_experiment/stratified_run_example/val/val.yaml"
with open(path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# %%
model, dataloader, loss_metric = evaluate(config)

# %%
print("Model:\n\n{}\n....\n".format(repr(model)[0:500]))

# %%
img_idx = 180
dataset = dataloader.dataset
x, y, index = dataset[img_idx]
# some losses depend on specific example data
extra_dict = dataset.get_extra_dict(index)

print("Dataset Length: {}".format(len(dataset)))
print("image -> shape {},  dtype {}".format(x.shape, x.dtype))
print("mask -> shape {},  dtype {}".format(y.shape, y.dtype))

# %%
print("Image and Mask: \n")
dcm, mask = x[0, ...], y

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(dcm, cmap="gray")
ax2.imshow(dcm, cmap="gray")
ax2.imshow(masks_to_colorimg(mask), alpha=0.5)

# %%
x, y, attribs = dataset.get_verbose(img_idx)
print("Image Attributes: \n\n{}".format(attribs))

# %%
print("Display Dataloader Batch")
data_iter = iter(dataloader)
for inputs, labels, index in data_iter:
    display_traindata(inputs[:12], labels[:12])
    extra_dict = dataset.get_extra_dict(index)
    break

# %%
out = model(inputs[:1])
reduced_extra_dict = {k: v[:1] for k, v in extra_dict.items()}
# %%
metric = loss_metric(out, labels[:1], reduced_extra_dict)
metric
# %%
