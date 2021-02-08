"""
Notebook to explore inference and metrics on patients

The makelinks flag is needed only once to create symbolic links to the data.
"""

# %%
from collections import OrderedDict, defaultdict

import pandas as pd
import cv2
import torch
import os
import yaml
import numpy as np
import json
from tqdm import tqdm
import nibabel as nib

import matplotlib.pyplot as plt
from matplotlib import cm
import albumentations
from torchvision.utils import make_grid

# from mpl_toolkits.mplot3d import Axes3D

# enable lib loading even if not installed as a pip package or in PYTHONPATH
# also convenient for relative paths in example config files
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

from adpkd_segmentation.config.config_utils import get_object_instance  # noqa
from adpkd_segmentation.datasets import dataloader as _dataloader  # noqa
from adpkd_segmentation.datasets import datasets as _datasets  # noqa
from adpkd_segmentation.data.link_data import makelinks  # noqa
from adpkd_segmentation.data.data_utils import display_sample  # noqa
from adpkd_segmentation.utils.train_utils import load_model_data  # noqa
from adpkd_segmentation.utils.stats_utils import (  # noqa
    bland_altman_plot,
    scatter_plot,
    linreg_plot,
)

from adpkd_segmentation.utils.losses import (
    SigmoidBinarize,
    Dice,
    binarize_thresholds,
)  # noqa
from torch.nn import Sigmoid

# %%


def load_config(config_path, run_makelinks=False):
    """Reads config file and calculates additional dcm attributes such as
    slice volume. Returns a dictionary used for patient wide calculations
    such as TKV.

    Args:
        config_path (str): config file path
        run_makelinks (bool, optional): Creates symbolic links during the first run. Defaults to False.

    Returns:
        dataloader, model, device, binarize_func, save_dir (str), model_name (str), split (str)
    """

    if run_makelinks:
        makelinks()
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_config = config["_MODEL_CONFIG"]
    loader_to_eval = config["_LOADER_TO_EVAL"]
    dataloader_config = config[loader_to_eval]
    saved_checkpoint = config["_MODEL_CHECKPOINT"]
    checkpoint_format = config["_NEW_CKP_FORMAT"]

    model = get_object_instance(model_config)()
    if saved_checkpoint is not None:
        load_model_data(saved_checkpoint, model, new_format=checkpoint_format)

    dataloader = get_object_instance(dataloader_config)()

    # TODO: support other metrics as needed
    # binarize_func = SigmoidBinarize(thresholds=[0.5])

    pred_process_config = config["_LOSSES_METRICS_CONFIG"]["criterions_dict"][
        "dice_metric"
    ]["pred_process"]
    pred_process = get_object_instance(pred_process_config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    model_name = Path(config_path).parts[-3]

    save_dir = "./saved_inference"

    return (dataloader, model, device, pred_process, save_dir, model_name)


def plot_model_results(csv_path, name):
    df = pd.read_csv(csv_path)
    pred = df["TKV_Pred"].to_numpy()
    gt = df["TKV_GT"].to_numpy()
    bland_altman_plot(
        pred, gt, percent=True, title=f"{name} BA Plot: TKV % error"
    )

    patient_dice = df["patient_dice"].to_numpy()
    scatter_plot(patient_dice, gt, title=f"{name} Dice by TKV")
    linreg_plot(pred, gt, title=f"{name} Linear Fit")


def inference_to_disk(
    dataloader,
    model,
    device,
    binarize_func,
    save_dir="./saved_inference",
    model_name="model",
):
    dataset = dataloader.dataset
    output_idx_check = (
        hasattr(dataloader.dataset, "output_idx")
        and dataloader.dataset.output_idx
    )

    assert (
        output_idx_check is True
    ), "output indexes are required for the dataset"

    for batch_idx, output in enumerate(dataloader):

        x_batch, idxs_batch = output
        x_batch = x_batch.to(device)

        with torch.no_grad():

            # get_verbose returns (sample, dcm_path, attributes dict)
            file_names = [
                Path(dataset.get_verbose(idx)[1]).stem for idx in idxs_batch
            ]

            file_attribs = [dataset.get_verbose(idx)[2] for idx in idxs_batch]

            y_batch_hat = model(x_batch)
            # TODO: support only sigmoid saves
            y_batch_hat_binary = binarize_func(y_batch_hat)

            for file_name, file_attrib, img, logit, pred in zip(
                file_names,
                file_attribs,
                x_batch,
                y_batch_hat,
                y_batch_hat_binary,
            ):
                out_dir = (
                    Path.cwd()
                    / Path(save_dir)
                    / model_name
                    / file_attrib["patient"]
                    / file_attrib["MR"]
                    / file_name
                )

                out_dir.parent.mkdir(parents=True, exist_ok=True)
                # print(out_dir)

                np.save(str(out_dir) + "_img", img.cpu().numpy())
                np.save(str(out_dir) + "_logit", logit.cpu().numpy())
                np.save(str(out_dir) + "_pred", pred.cpu().numpy())

                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        else:
                            return super(NpEncoder, self).default(obj)

                # get resize transform within compose object
                Resize = albumentations.augmentations.transforms.Resize
                transform_resize = next(
                    v
                    for v in dataloader.dataset.augmentation.transforms
                    if isinstance(v, Resize)
                )
                assert (
                    transform_resize is not None
                ), "transform_resize must be defined"

                file_attrib["transform_resize_dim"] = (
                    transform_resize.height,
                    transform_resize.width,
                )

                attrib_json = json.dumps(file_attrib, cls=NpEncoder)
                f = open(str(out_dir) + "_attrib.json", "w")
                f.write(attrib_json)
                f.close()


# %%
def resized_stack(numpy_list, dsize=None):
    """resizing lists of array with dimension:
    slices x 1 x H x W, where H = W.

    Sets output size to first array at idx 0 or dsize

    Args:
        numpy_list (list): list of numpy arr
        dsize (int, optional): output dimension. Defaults to None.

    Returns:
        numpy: stacked numpy lists with same size
    """
    assert numpy_list[0].shape[1] == 1, "dimension check"
    assert numpy_list[0].shape[2] == numpy_list[0].shape[3], "square check"

    def reshape(arr):
        """reshapes [slices x 1 x H x W] to [H x W x slices]"""
        arr = np.moveaxis(arr, 0, -1)  # slices to end
        arr = np.squeeze(arr)  # remove 1 dimension
        return arr

    reshaped = [reshape(arr) for arr in numpy_list]

    if dsize is None:
        dsize = reshaped[0].shape[0:2]  # get H, W from first arr

    resized = [
        cv2.resize(src, dsize, interpolation=cv2.INTER_CUBIC)
        for src in reshaped
    ]

    return np.stack(resized)


def display_volumes(
    study_dir,
    style="prob",
    plot_error=False,
    skip_display=True,
    save_dir=None,
):

    print(f"loading from {study_dir}")
    study_dir = Path(study_dir)
    imgs = sorted(study_dir.glob("*_img.npy"))
    imgs_np = [np.load(i) for i in imgs]
    logits = sorted(study_dir.glob("*_logit.npy"))
    logits_np = [np.load(logit) for logit in logits]
    preds = sorted(study_dir.glob("*_pred.npy"))
    preds_np = [np.load(p) for p in preds]

    vols = {
        "img": np.stack(imgs_np),
        "logit": np.stack(logits_np),
        "pred": np.stack(preds_np),
        "prob": torch.sigmoid(torch.from_numpy(np.stack(logits_np))).numpy(),
    }

    def show(img, label=None, img_alpha=1, lb_alpha=0.3):
        npimg = img.numpy()
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(
            np.transpose(npimg, (1, 2, 0)),
            interpolation="none",
            alpha=img_alpha,
        )
        if label is not None:
            lbimg = label.numpy()
            ax.imshow(
                np.transpose(lbimg, (1, 2, 0)),
                alpha=lb_alpha,
                interpolation="none",
            )

    x = torch.from_numpy(vols["img"])
    y = vols[style]

    def norm_tensor(x):
        x = x / x.sum(0).expand_as(x)
        x[torch.isnan(x)] = 0
        return x

    bkgrd_thresh = 0.01
    cmap_vol = np.ma.masked_where(y <= bkgrd_thresh, y)
    cmap_vol = np.apply_along_axis(cm.inferno, 0, cmap_vol)
    cmap_vol = torch.from_numpy(np.squeeze(cmap_vol))

    print(f"style is: {style}")
    print(f"vol stats: min:{y.min()} max:{y.max()} mean:{y.mean()}")

    if not skip_display:
        if style is "img":
            show(make_grid(x), lb_alpha=0.5)
            plt.show()
        else:
            show(make_grid(x), make_grid(cmap_vol), lb_alpha=0.5)
            if save_dir is None:
                plt.show()
            else:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(Path(save_dir) / "label_grid.svg")
                plt.savefig(Path(save_dir) / "label_grid.png")
    return y


def exam_preds_to_stat(
    pred_vol, ground_vol, pred_process, attrib_dict, pred_std=None
):
    """computes stats for a single exam prediction

    Args:
        pred_vol (numpy): prediction volume
        ground_vol (numpy): ground truth volume
        pred_process (function): converts prediction to binary
        attrib (dict): dictionary of attributes (usually from index 0)

    Returns:
        tuple: study key, dictionary of attributes
    """
    volume_ground = None
    volume_pred = None
    dice = Dice(
        pred_process=pred_process, use_as_loss=False, power=1, dim=(0, 1, 2, 3)
    )
    dice_val = dice(
        torch.from_numpy(pred_vol), torch.from_numpy(ground_vol)
    ).item()

    scale_factor = (attrib_dict["dim"][0] ** 2) / (
        attrib_dict["transform_resize_dim"][0] ** 2
    )
    # print(f"scale factor {scale_factor}")
    pred_pixel_count = torch.sum(
        pred_process(torch.from_numpy(pred_vol))
    ).item()
    volume_pred = scale_factor * attrib_dict["vox_vol"] * pred_pixel_count

    ground_pixel_count = torch.sum(
        pred_process(torch.from_numpy(ground_vol))
    ).item()
    volume_ground = scale_factor * attrib_dict["vox_vol"] * ground_pixel_count

    attrib_dict.update(
        {
            "TKV_GT": volume_ground,
            "TKV_Pred": volume_pred,
            "patient_dice": dice_val,
            "study": attrib_dict["patient"] + attrib_dict["MR"],
            "scale_factor": scale_factor,
            "Pred_stdev": pred_std,
        }
    )

    return attrib_dict


def compute_inference_stats(
    save_dir, output=False, display=False, patient_ID=None
):

    Metric_data = OrderedDict()
    Combined_metric_data = OrderedDict()
    root = Path.cwd() / Path(save_dir)

    model_inferences = list(root.glob("*"))
    newline = "\n"
    formated_list = "".join([f"{newline} {m}" for m in model_inferences])

    print(f"calculating model inferences for {formated_list}")

    all_logit_vol = defaultdict(list)
    all_pred_vol = defaultdict(list)
    all_ground_vol = defaultdict(list)
    all_summaries = defaultdict(list)

    pred_process = SigmoidBinarize(thresholds=[0.5])

    for model_dir in tqdm(model_inferences):
        if patient_ID is not None:
            MR_num = "*"
        else:
            patient_ID, MR_num = "*", "*"
        studies = model_dir.glob(f"{patient_ID}/{MR_num}")

        for study_dir in studies:
            imgs = sorted(study_dir.glob("*_img.npy"))
            imgs_np = [np.load(i) for i in imgs]
            logits = sorted(study_dir.glob("*_logit.npy"))
            logits_np = [np.load(logit) for logit in logits]
            preds = sorted(study_dir.glob("*_pred.npy"))
            preds_np = [np.load(p) for p in preds]
            grounds = sorted(study_dir.glob("*_ground.npy"))
            grounds_np = [np.load(g) for g in grounds]
            attribs = sorted(study_dir.glob("*_attrib.json"))
            attribs_dicts = []
            for a in attribs:
                with open(a) as json_file:
                    attribs_dicts.append(json.load(json_file))

            # volumes for a study within one model inference
            img_vol = np.stack(imgs_np)
            logit_vol = np.stack(logits_np)
            pred_vol = np.stack(preds_np)
            ground_vol = np.stack(grounds_np)

            if display is True:
                display_volumes(img_vol, pred_vol, ground_vol)

            summary = exam_preds_to_stat(
                pred_vol, ground_vol, pred_process, attribs_dicts[0]
            )

            Metric_data[summary["study"]] = summary

            # accumulate predictions across all models for each study
            all_logit_vol[summary["study"]].append(logit_vol)
            all_pred_vol[summary["study"]].append(pred_vol)
            all_ground_vol[summary["study"]].append(ground_vol)
            all_summaries[summary["study"]].append(summary)

        df = pd.DataFrame(Metric_data).transpose()

        if output is True:
            df.to_csv(f"stats-{model_dir.name}.csv")

    for key, value in all_logit_vol.items():
        # uses index 0 to get ground truth and standard voxel attribs

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # resizes by index 0
        prob_vol = resized_stack(value)
        # prob_vol = sigmoid(prob_vol)
        prob_vol = np.mean(prob_vol, axis=0)
        prob_std = np.std(prob_vol)

        prob_vol = np.moveaxis(prob_vol, -1, 0)  # b x (X x Y)
        prob_vol = np.expand_dims(prob_vol, axis=1)  # b x c x (X x Y)
        pred_vol = binarize_thresholds(torch.from_numpy(prob_vol)).numpy()
        ground_vol = all_ground_vol[key][0]

        summary = exam_preds_to_stat(
            pred_vol,
            ground_vol,
            pred_process,
            all_summaries[key][0],
            pred_std=prob_std,
        )

        Combined_metric_data[summary["study"]] = summary

    df = pd.DataFrame(Combined_metric_data).transpose()

    if output is True:
        print("saving combined csv")
        df.to_csv("stats-combined_models.csv")


# %%
# Single Experiment
# path = "./experiments/november/26_new_stratified_run_2_long_512/test/test.yaml"

# Ensemble Experiment
paths = [
    "adpkd-segmentation/checkpoints/inference.yml",
]

# %%
# single inference
# *model_args, split = load_config(config_path=path)


# %%
# multi-model inference
for p in tqdm(paths):
    model_args = load_config(config_path=p)
    inference_to_disk(*model_args)

# %%
save_dir = "adpkd-segmentation/saved_inference/adpkd-segmentation/1001707280/Axial T2 SS-FSE"
y = display_volumes(
    study_dir="adpkd-segmentation/saved_inference/adpkd-segmentation/1001707280/Axial T2 SS-FSE",
    style="pred",
    plot_error=True,
    skip_display=False,
    save_dir=save_dir,
)
# %%
files = Path("saved_inference").glob("**/*")
folders = [f.parent for f in files if len(f.parent.parts) == 4]
folders = list(set(folders))
saved_folders = [
    Path("saved_figs") / f"{d.parts[-2]}_{d.parts[-1]}" for d in folders
]
# %%

for study_dir, save_dir in tqdm(list(zip(folders, saved_folders))[17:]):

    try:
        y = display_volumes(
            study_dir=study_dir,
            style="pred",
            plot_error=True,
            skip_display=False,
            save_dir=save_dir,
        )
    except:
        pass
# %%

saved_inference = "adpkd-segmentation/saved_inference"
files = list(Path(saved_inference).glob("**/*img.npy"))
folders = list(set([f.parent for f in files]))
print(folders)

# %%
from ast import literal_eval
import pydicom


for folder in folders:
    imgs = list(folder.glob("*img.npy"))
    imgs = sorted(imgs)
    np_imgs = [(np.load(i))[0] for i in imgs]
    np_vol = np.stack(np_imgs).T
    print(np_vol.shape)
    print(f"{folder.name}_img_vol.nii")
    nifi_vol = nib.Nifti1Image(np_vol, affine=np.eye(4))
    nib.save(nifi_vol, folder / f"{folder.name}_img_vol.nii")

# %%
