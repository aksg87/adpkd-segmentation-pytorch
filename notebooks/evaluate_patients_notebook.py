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

import matplotlib.pyplot as plt

# enable lib loading even if not installed as a pip package or in PYTHONPATH
# also convenient for relative paths in example config files
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

from adpkd_segmentation.config.config_utils import get_object_instance  # noqa
from adpkd_segmentation.data.link_data import makelinks  # noqa
from adpkd_segmentation.data.data_utils import display_sample  # noqa
from adpkd_segmentation.utils.train_utils import load_model_data  # noqa
from adpkd_segmentation.utils.stats_utils import (  # noqa
    bland_altman_plot,
    scatter_plot,
    linreg_plot,
)

from adpkd_segmentation.utils.losses import SigmoidBinarize, Dice  # noqa
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
    split = config[loader_to_eval]["dataset"]["splitter_key"].lower()
    dataloader_config = config[loader_to_eval]
    saved_checkpoint = config["_MODEL_CHECKPOINT"]
    checkpoint_format = config["_NEW_CKP_FORMAT"]

    model = get_object_instance(model_config)()
    if saved_checkpoint is not None:
        load_model_data(saved_checkpoint, model, new_format=checkpoint_format)

    dataloader = get_object_instance(dataloader_config)()

    # TODO: support other metrics as needed
    binarize_func = SigmoidBinarize(thresholds=[0.5])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    model_name = Path(config_path).parts[-3]

    save_dir = "./saved_inference"

    return (
        dataloader,
        model,
        device,
        binarize_func,
        save_dir,
        model_name,
        split,
    )


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
    output_example_idx = (
        hasattr(dataloader.dataset, "output_idx")
        and dataloader.dataset.output_idx
    )

    for batch_idx, output in enumerate(dataloader):
        if output_example_idx:
            x_batch, y_batch, idxs_batch = output
        else:
            x_batch, y_batch = output

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():

            file_names = [
                Path(dataset.get_verbose(idx)[1]).stem for idx in idxs_batch
            ]

            file_attribs = [dataset.get_verbose(idx)[2] for idx in idxs_batch]

            y_batch_hat = model(x_batch)
            y_batch_hat_binary = binarize_func(y_batch_hat)

            for file_name, file_attrib, img, pred, ground in zip(
                file_names, file_attribs, x_batch, y_batch_hat_binary, y_batch
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
                np.save(str(out_dir) + "_pred", pred.cpu().numpy())
                np.save(str(out_dir) + "_ground", ground.cpu().numpy())

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

                trans_resize = dataloader.dataset.augmentation[0]
                file_attrib["transform_resize_dim"] = (
                    trans_resize.height,
                    trans_resize.width,
                )

                attrib_json = json.dumps(file_attrib, cls=NpEncoder)
                f = open(str(out_dir) + "_attrib.json", "w")
                f.write(attrib_json)
                f.close()


# %%
def resized_stack(numpy_list, dsize=None):
    """resizing lists of array with dimension:
    slices x 1 x H' x W'.

    H' x W' default to the first array or dsize
    """
    assert numpy_list[0].shape[1] == 1, "dimension check"
    assert numpy_list[0].shape[2] == numpy_list[0].shape[3], "dimension check"

    def reshape(arr):
        """reshapes to dimension:
        H x W x slices
        """
        arr = np.moveaxis(arr, 0, -1)
        arr = np.squeeze(arr)
        return arr

    reshaped = [reshape(arr) for arr in numpy_list]

    if dsize is None:
        dsize = reshaped[0].shape[0:2]

    resized = [
        cv2.resize(src, dsize, interpolation=cv2.INTER_CUBIC)
        for src in reshaped
    ]

    return np.stack(resized)


def compute_inference_stats(
    save_dir="./saved_inference", output=False, display=False
):

    Metric_data = OrderedDict()
    Combined_metric_data = OrderedDict()
    root = Path.cwd() / Path(save_dir)

    model_inferences = list(root.glob("*"))
    newline = "\n"
    formated_list = "".join([f"{newline} {m}" for m in model_inferences])

    print(f"calculating model inferences for {formated_list}")

    all_pred_vol = defaultdict(list)
    all_ground_vol = defaultdict(list)
    all_summaries = defaultdict(list)

    pred_process = SigmoidBinarize(thresholds=[0.5])
    dice = Dice(
        pred_process=pred_process, use_as_loss=False, power=1, dim=(0, 1, 2, 3)
    )

    for model_dir in tqdm(model_inferences):
        patient_ID, MR_num = "*", "*"
        studies = model_dir.glob(f"{patient_ID}/{MR_num}")

        for study_dir in studies:
            imgs = sorted(study_dir.glob("*_img.npy"))
            imgs_np = [np.load(i) for i in imgs]

            preds = sorted(study_dir.glob("*_pred.npy"))
            preds_np = [np.load(p) for p in preds]

            grounds = sorted(study_dir.glob("*_ground.npy"))
            grounds_np = [np.load(g) for g in grounds]

            attribs = sorted(study_dir.glob("*_attrib.json"))

            # volumes for a study within one model inference
            img_vol = np.stack(imgs_np)
            pred_vol = np.stack(preds_np)
            ground_vol = np.stack(grounds_np)

            # accumulate volumes across all models for each study
            rel_path = study_dir.relative_to(model_dir)
            all_pred_vol[rel_path].append(pred_vol)
            all_ground_vol[rel_path].append(ground_vol)

            if display is True:
                slice1 = 1 * (pred_vol.shape[0] // 6)
                slice2 = 2 * (pred_vol.shape[0] // 6)
                slice3 = 3 * (pred_vol.shape[0] // 6)
                slice4 = 4 * (pred_vol.shape[0] // 6)
                slice5 = 5 * (pred_vol.shape[0] // 6)

                f, ax = plt.subplots(5, 2)

                ax[0, 0].imshow(img_vol[slice1, 0], cmap="gray")
                ax[0, 1].imshow(img_vol[slice1, 0], cmap="gray")
                ax[1, 0].imshow(img_vol[slice2, 0], cmap="gray")
                ax[1, 1].imshow(img_vol[slice2, 0], cmap="gray")
                ax[2, 0].imshow(img_vol[slice3, 0], cmap="gray")
                ax[2, 1].imshow(img_vol[slice3, 0], cmap="gray")
                ax[3, 0].imshow(img_vol[slice4, 0], cmap="gray")
                ax[3, 1].imshow(img_vol[slice4, 0], cmap="gray")
                ax[4, 0].imshow(img_vol[slice5, 0], cmap="gray")
                ax[4, 1].imshow(img_vol[slice5, 0], cmap="gray")

                ax[0, 0].imshow(pred_vol[slice1, 0], cmap="viridis", alpha=0.3)
                ax[0, 1].imshow(
                    ground_vol[slice1, 0], cmap="viridis", alpha=0.3
                )
                ax[1, 0].imshow(pred_vol[slice2, 0], cmap="viridis", alpha=0.3)
                ax[1, 1].imshow(
                    ground_vol[slice2, 0], cmap="viridis", alpha=0.3
                )
                ax[2, 0].imshow(pred_vol[slice3, 0], cmap="viridis", alpha=0.3)
                ax[2, 1].imshow(
                    ground_vol[slice3, 0], cmap="viridis", alpha=0.3
                )
                ax[3, 0].imshow(pred_vol[slice4, 0], cmap="viridis", alpha=0.3)
                ax[3, 1].imshow(
                    ground_vol[slice4, 0], cmap="viridis", alpha=0.3
                )
                ax[4, 0].imshow(pred_vol[slice5, 0], cmap="viridis", alpha=0.3)
                ax[4, 1].imshow(
                    ground_vol[slice5, 0], cmap="viridis", alpha=0.3
                )

                f.tight_layout()

            dice_val = dice(
                torch.from_numpy(pred_vol), torch.from_numpy(ground_vol)
            ).item()

            volume_ground = None
            volume_pred = None
            with open(attribs[0]) as json_file:
                attrib = json.load(json_file)

                scale_factor = (attrib["dim"][0] ** 2) / (
                    attrib["transform_resize_dim"][0] ** 2
                )
                # print(f"scale factor {scale_factor}")
                pred_pixel_count = torch.sum(
                    pred_process(torch.from_numpy(pred_vol))
                ).item()
                volume_pred = (
                    scale_factor * attrib["vox_vol"] * pred_pixel_count
                )

                ground_pixel_count = torch.sum(
                    pred_process(torch.from_numpy(ground_vol))
                ).item()
                volume_ground = (
                    scale_factor * attrib["vox_vol"] * ground_pixel_count
                )

                # print(f"volume_pred:{volume_pred}  volume_ground:{volume_ground}")

                summary = {
                    "TKV_GT": volume_ground,
                    "TKV_Pred": volume_pred,
                    "sequence": attrib["seq"],
                    "split": split,
                    "patient_dice": dice_val,
                    "study": attrib["patient"] + attrib["MR"],
                    "scale_factor": scale_factor,
                    "vox_vol": attrib["vox_vol"],
                    "dim": attrib["dim"][0],
                    "transform_resize_dim": attrib["transform_resize_dim"][0],
                }

                study = attrib["patient"] + attrib["MR"]
                Metric_data[study] = summary

                all_summaries[rel_path].append(summary)

        df = pd.DataFrame(Metric_data).transpose()

        if output is True:
            df.to_csv(f"stats-{model_dir.name}.csv")

    for key, value in all_pred_vol.items():
        all_pred_vol[key] = resized_stack(value)
        summary = all_summaries[key][0]
        pred_vol = np.mean(all_pred_vol[key], axis=0)
        pred_std = np.std(all_pred_vol[key])

        # move back to slices x 1 x H x W
        pred_vol = np.moveaxis(pred_vol, -1, 0)
        pred_vol = np.expand_dims(pred_vol, axis=1)

        # res.append(pred_vol)
        # return res
        ground_vol = all_ground_vol[key][0]

        # print(f"scale factor {scale_factor}")
        pred_pixel_count = torch.sum(
            pred_process(torch.from_numpy(pred_vol))
        ).item()
        scale_factor = summary["scale_factor"]
        vox_vol = summary["vox_vol"]

        volume_pred = scale_factor * vox_vol * pred_pixel_count
        ground_pixel_count = torch.sum(
            pred_process(torch.from_numpy(ground_vol))
        ).item()
        volume_ground = scale_factor * vox_vol * ground_pixel_count

        dice_val = dice(
            torch.from_numpy(pred_vol), torch.from_numpy(ground_vol)
        ).item()

        summary["TKV_Pred"] = volume_pred
        summary["TKV_GT"] = volume_ground
        summary["patient_dice"] = dice_val
        summary["Pred_stdev"] = pred_std

        Combined_metric_data[summary["study"]] = summary

    df = pd.DataFrame(Combined_metric_data).transpose()

    if output is True:
        df.to_csv(f"stats-combined_models.csv")


# %%
path = "./experiments/november/26_new_stratified_run_2_long_512/test/test.yaml"

paths = [
    "./experiments/november/25_new_stratified_run_1/test/test.yaml",
    "./experiments/november/25_new_stratified_run_2/test/test.yaml",
    "./experiments/november/25_new_stratified_run_2_long/test/test.yaml",
    "./experiments/november/26_new_stratified_run_2/test/test.yaml",
    "./experiments/november/26_new_stratified_run_2_long/test/test.yaml",
    "./experiments/november/26_new_stratified_run_2_long_512/test/test.yaml",
    "./experiments/november/26_new_stratified_run_2_long_512_b6/test/test.yaml",
    "./experiments/november/26_new_stratified_run_2_long_advprop/test/test.yaml",
    "./experiments/november/26_new_stratified_run_2_long_batchdice1/test/test.yaml",
    "./experiments/november/26_new_stratified_run_2_long_noisy-student/test/test.yaml",
]

# %%
# single inference
*model_args, split = load_config(config_path=path)
all_loaded_configs = [load_config(config_path=p) for p in paths]


# %%
# multi-model inference
for loaded_configs in tqdm(all_loaded_configs):
    *model_args, split = loaded_configs
    inference_to_disk(*model_args)

# %%
# run calculations on all saved inferences
compute_inference_stats(output=True)

# %%
# make plot for all saved stats
stats_csvs = list(Path.cwd().glob("stats-*"))

for csv_f in stats_csvs:
    plot_model_results(csv_f, csv_f.name)

# %%