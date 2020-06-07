"""
Model evaluation script

python -m evaluate --config path_to_config_yaml
"""

# %%
import yaml

import matplotlib.pyplot as plt

from config.config_utils import get_object_instance
from data.link_data import makelinks
from data.data_utils import masks_to_colorimg, display_traindata

# %%
makelinks()


# %%
def evaluate(config):
    model_config = config["_MODEL_CONFIG"]
    loss_criterion_config = config["_LOSS_CRITERION_CONFIG"]
    dataloader_config = config["_VAL_DATALOADER_CONFIG"]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]

    model = get_object_instance(model_config)()
    loss_criterion = get_object_instance(loss_criterion_config)
    dataloader = get_object_instance(dataloader_config)()
    loss_metric = get_object_instance(loss_metric_config)()

    return (
        model,
        loss_criterion,
        dataloader,
        loss_metric,
    )  # add return types for debugging/testing


# %%
path = "./config/examples/eval_example.yaml"

# %%
with open(path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# %%
model, loss_criterion, dataloader, loss_metric = evaluate(config)

# %%
print("Model:\n\n{}\n....\n".format(repr(model)[0:500]))

# %%
print("Loss: {}".format(loss_criterion))

# %%
img_idx = 180

dataset = dataloader.dataset
x, y = dataset[img_idx]
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
# %%
for inputs, labels in data_iter:

    display_traindata(inputs[:12], labels[:12])
    break

# %%
loss_metric(labels, model(inputs))

# %%
