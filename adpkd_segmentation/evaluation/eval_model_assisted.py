from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

from argparse import ArgumentParser
import pprint
from pydicom import dcmread
import nibabel as nib

from adpkd_segmentation.evaluation.evaluation_utils import exam_preds_to_stat

MODEL = "model"
MODEL_ASSISTED = "model_assisted"
VOX_VOL = "vox_vol"
STUDY = "study"

parser = ArgumentParser()
pp = pprint.PrettyPrinter(indent=4)

parser.add_argument(
    "-i",
    "--input_path",
    type=str,
    help="path to model predictions and model_assisted_annotation",
    default="/big_data2/apkd_segmentation/storage/output/saved_inference/adpkd-segmentation-pytorch",
)

parser.add_argument(
    "-o",
    "--output_path",
    type=str,
    help="path to output dictory",
    default="/home/akg9006/test_model_assisted",
)


def pred_ground_paths_dict(input_path, ground_keys=["a_vol.nii", "a.nii"]):
    """Returns a dictionary with prediction, ground, and voxel volume."""

    folder_to_data = defaultdict(dict)

    model_pred = list(input_path.glob("**/pred_vol.nii"))
    model_assisted = []
    for gk in ground_keys:
        model_assisted.extend(list(input_path.glob(f"**/{gk}")))

    for mp in model_pred:
        folder_to_data[mp.parent][MODEL] = mp
    for ma in model_assisted:
        assert (
            MODEL_ASSISTED not in folder_to_data[ma.parent]
        ), f"Only one ground_truth volume per folder. Check {ma.parent}"
        folder_to_data[ma.parent][MODEL_ASSISTED] = ma

    folder_to_data = {
        k: v
        for k, v in folder_to_data.items()
        if MODEL in v and MODEL_ASSISTED in v
    }

    for k, v in folder_to_data.items():
        # extract voxel_volume from some dicom in parent folder
        dcms = list(k.parent.glob("*.dcm"))
        assert (
            len(dcms) > 0
        ), f"Dicom files required in {k.parent} for volume calculations"
        if len(dcms) > 0:
            pdcm = dcmread(dcms[0])
            dX_Y = float(pdcm.PixelSpacing[0])
            dZ = float(pdcm.SpacingBetweenSlices)
            vox_vol = dZ * (dX_Y ** 2)
            # print(f"DCM: {dcms[0].name}  vox_vol: {vox_vol}")
            v[VOX_VOL] = vox_vol

    for k, v in folder_to_data.items():
        v[STUDY] = str(v[MODEL].parent.relative_to(input_path))

    return folder_to_data


def compute_pred_vs_ground(
    folder_to_date, output=None, exclude_keys=["scale_factor", "Pred_stdev"]
):
    """Consumes dictionary with paths to model, model_assisted and
    data for voxel_volume. Computes dice and TKV calculations.

    Args:
        folder_to_date (dict): Contains keys MODEL, MODEL_ASSITED, VOX_VOLUME

    Returns:
        pd.DataFrame: Contains results for each study
    """

    all_stats = defaultdict(list)
    for k, v in folder_to_data.items():
        pred_vol = nib.load(v[MODEL]).get_fdata()
        pred_vol = np.expand_dims(pred_vol, axis=0)

        ground_vol = nib.load(v[MODEL_ASSISTED]).get_fdata()
        ground_vol = np.expand_dims(ground_vol, axis=0)

        vox_vol = v[VOX_VOL]
        study = v[STUDY]

        stat = exam_preds_to_stat(
            pred_vol=pred_vol,
            ground_vol=ground_vol,
            vox_vol=vox_vol,
            study=study,
        )

        for k, v in stat.items():
            if k not in exclude_keys:
                all_stats[k].append(v)

    df = pd.DataFrame.from_dict(all_stats)

    if output is not None:
        if output.is_dir():
            output = output / "model_assisted_vs_model.csv"
        df.to_csv(output)

    return df


if __name__ == "__main__":
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    folder_to_data = pred_ground_paths_dict(input_path)

    print(f"Generating stats. Input: {input_path} Output: {output_path}")
    all_stats = compute_pred_vs_ground(folder_to_data, output_path)
    print(all_stats)
