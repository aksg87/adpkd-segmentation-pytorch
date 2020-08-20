# Notebook to check different augmentations

# %%
import matplotlib.pyplot as plt
import os
import yaml

# https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_kaggle_salt.ipynb # noqa
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    RandomResizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate,
    IAASharpen,
    Blur,
    MotionBlur,
    ImageCompression,
    IAAPerspective,
    MultiplicativeNoise,
)

# enable lib loading even if not installed as a pip package or in PYTHONPATH
# also convenient for relative paths in example config files
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

from adpkd_segmentation.config.config_utils import get_object_instance  # noqa
from adpkd_segmentation.data.link_data import makelinks  # noqa
from adpkd_segmentation.data.data_utils import ( # noqa
    int16_to_uint8,
    masks_to_colorimg,
)

# %%
# needed only once
# makelinks()

# %%
path = "./misc/example_experiment/stratified_run_example/val/val.yaml"
with open(path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
dataloader_config = config["_VAL_DATALOADER_CONFIG"]
dataloader = get_object_instance(dataloader_config)()

# %%
# SET THIS INDEX for selecting img label in augmentations example
IMG_IDX = 180
dataset = dataloader.dataset
x, y, index = dataset[IMG_IDX]

# %%
print("Dataset Length: {}".format(len(dataset)))
print("image -> shape {},  dtype {}".format(x.shape, x.dtype))
print("mask -> shape {},  dtype {}".format(y.shape, y.dtype))

# %%
print("Image and Mask: \n")
image, mask = x[0, ...], y

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image, cmap="gray")
ax2.imshow(image, cmap="gray")
ax2.imshow(masks_to_colorimg(mask), alpha=0.5)


# %%
# from albumentation examples
def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title("Original image", fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title("Original mask", fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title("Transformed image", fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title("Transformed mask", fontsize=fontsize)


# %%
# ORIGINAL
mask = mask[0]
visualize(image, mask)

# %%
# PADDING
aug = PadIfNeeded(p=1, min_height=256, min_width=256)

augmented = aug(image=image, mask=mask)

image_padded = augmented["image"]
mask_padded = augmented["mask"]

print(image_padded.shape, mask_padded.shape)

visualize(image_padded, mask_padded, original_image=image, original_mask=mask)

# %%
# CENTER CROP
original_height, original_width = 224, 224

aug = CenterCrop(p=1, height=original_height, width=original_width)

augmented = aug(image=image_padded, mask=mask_padded)

image_center_cropped = augmented["image"]
mask_center_cropped = augmented["mask"]

print(image_center_cropped.shape, mask_center_cropped.shape)

assert (image - image_center_cropped).sum() == 0
assert (mask - mask_center_cropped).sum() == 0

visualize(
    image_center_cropped,
    mask_center_cropped,
    original_image=image_padded,
    original_mask=mask_padded,
)


# %%
# Horizontal Flip
aug = HorizontalFlip(p=1)

augmented = aug(image=image, mask=mask)

image_h_flipped = augmented["image"]
mask_h_flipped = augmented["mask"]

visualize(
    image_h_flipped, mask_h_flipped, original_image=image, original_mask=mask
)

# %%

# Vertical Flip
aug = VerticalFlip(p=1)

augmented = aug(image=image, mask=mask)

image_v_flipped = augmented["image"]
mask_v_flipped = augmented["mask"]

visualize(
    image_v_flipped, mask_v_flipped, original_image=image, original_mask=mask
)

# %%

# RandomRotate90  (Randomly rotates by 0, 90, 180, 270 degrees)
aug = RandomRotate90(p=1)

augmented = aug(image=image, mask=mask)

image_rot90 = augmented["image"]
mask_rot90 = augmented["mask"]

visualize(image_rot90, mask_rot90, original_image=image, original_mask=mask)
# %%

# Transpose (switch X and Y axis)
aug = Transpose(p=1)

augmented = aug(image=image, mask=mask)

image_transposed = augmented["image"]
mask_transposed = augmented["mask"]

visualize(
    image_transposed, mask_transposed, original_image=image, original_mask=mask
)

# %%

# ElasticTransform
aug = ElasticTransform(
    p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
)

augmented = aug(image=image, mask=mask)

image_elastic = augmented["image"]
mask_elastic = augmented["mask"]

visualize(
    image_elastic, mask_elastic, original_image=image, original_mask=mask
)

# %%

# ElasticTransform default
aug = ElasticTransform()

augmented = aug(image=image, mask=mask)

image_elastic = augmented["image"]
mask_elastic = augmented["mask"]

visualize(
    image_elastic, mask_elastic, original_image=image, original_mask=mask
)

# %%

# GridDistortion
aug = GridDistortion(distort_limit=0.3, p=1)

augmented = aug(image=image, mask=mask)

image_grid = augmented["image"]
mask_grid = augmented["mask"]

visualize(image_grid, mask_grid, original_image=image, original_mask=mask)

# %%
# Optical Distortion
aug = OpticalDistortion(p=1, distort_limit=1, shift_limit=0.3)

augmented = aug(image=image, mask=mask)

image_optical = augmented["image"]
mask_optical = augmented["mask"]

visualize(
    image_optical, mask_optical, original_image=image, original_mask=mask
)

# %%
# Optical Distortion 3
aug = OpticalDistortion(p=1, distort_limit=1, shift_limit=0.3)

augmented = aug(image=image, mask=mask)

image_optical = augmented["image"]
mask_optical = augmented["mask"]

visualize(
    image_optical, mask_optical, original_image=image, original_mask=mask
)

# %%
# Optical Distortion default
aug = OpticalDistortion(p=1)

augmented = aug(image=image, mask=mask)

image_optical = augmented["image"]
mask_optical = augmented["mask"]

visualize(
    image_optical, mask_optical, original_image=image, original_mask=mask
)

# %%
# Shift scale rotate
aug = ShiftScaleRotate(
    border_mode=0, rotate_limit=20, scale_limit=0.3, shift_limit=0.1
)

augmented = aug(image=image, mask=mask)

image_optical = augmented["image"]
mask_optical = augmented["mask"]

visualize(
    image_optical, mask_optical, original_image=image, original_mask=mask
)

# %%

# RandomSizedCrop

aug = RandomSizedCrop(p=1, min_max_height=(100, 200), height=128, width=128)

augmented = aug(image=image, mask=mask)

image_scaled = augmented["image"]
mask_scaled = augmented["mask"]

visualize(image_scaled, mask_scaled, original_image=image, original_mask=mask)

# %%

# RandomResizedCrop

aug = RandomResizedCrop(p=1, height=72, width=72, scale=(0.25, 1.0))

augmented = aug(image=image, mask=mask)

image_scaled = augmented["image"]
mask_scaled = augmented["mask"]

visualize(image_scaled, mask_scaled, original_image=image, original_mask=mask)


# %%
# CLAHE
aug = CLAHE()
image8 = (image * 256).astype("uint8")
augmented = aug(image=image8, mask=mask.astype("uint8"))

image_scaled = augmented["image"]
mask_scaled = augmented["mask"]

visualize(image_scaled, mask_scaled, original_image=image8, original_mask=mask)


# %%
# RandomBrightnessContrast
aug = RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)
image8 = (image * 256).astype("uint8")
augmented = aug(image=image8, mask=mask)

image_scaled = augmented["image"]
mask_scaled = augmented["mask"]

visualize(image_scaled, mask_scaled, original_image=image8, original_mask=mask)


# %%
# RandomGamma

aug = RandomGamma(gamma_limit=(40, 200))
image8 = (image * 256).astype("uint8")
augmented = aug(image=image8, mask=mask)

image_scaled = augmented["image"]
mask_scaled = augmented["mask"]

visualize(image_scaled, mask_scaled, original_image=image8, original_mask=mask)


# %%
# IAASharpen
aug = IAASharpen(alpha=(0.1, 0.2), lightness=(0.5, 0.7))
image8 = (image * 256).astype("uint8")
augmented = aug(image=image8, mask=mask)

image_scaled = augmented["image"]
mask_scaled = augmented["mask"]

visualize(image_scaled, mask_scaled, original_image=image8, original_mask=mask)


# %%
# Blur
aug = Blur(blur_limit=2)
image8 = (image * 256).astype("uint8")
augmented = aug(image=image8, mask=mask)

image_scaled = augmented["image"]
mask_scaled = augmented["mask"]

visualize(image_scaled, mask_scaled, original_image=image8, original_mask=mask)


# %%
# Motion Blur
aug = MotionBlur(blur_limit=5)
image8 = (image * 256).astype("uint8")
augmented = aug(image=image8, mask=mask)

image_scaled = augmented["image"]
mask_scaled = augmented["mask"]

visualize(image_scaled, mask_scaled, original_image=image8, original_mask=mask)


# %%
# Image Compression
aug = ImageCompression(quality_lower=50, quality_upper=50)
image8 = (image * 256).astype("uint8")
augmented = aug(image=image8, mask=mask)

image_scaled = augmented["image"]
mask_scaled = augmented["mask"]

visualize(image_scaled, mask_scaled, original_image=image8, original_mask=mask)


# %%
# IAAPerspective
aug = IAAPerspective()
image8 = (image * 256).astype("uint8")
mask8 = mask.astype("uint8")
augmented = aug(image=image8, mask=mask8)

image_scaled = augmented["image"]
mask_scaled = augmented["mask"]

visualize(
    image_scaled, mask_scaled, original_image=image8, original_mask=mask8
)


# %%
# MultiplicativeNoise
aug = MultiplicativeNoise(multiplier=(0.8, 1.2))
image8 = (image * 256).astype("uint8")
mask8 = mask.astype("uint8")
augmented = aug(image=image8, mask=mask8)

image_scaled = augmented["image"]
mask_scaled = augmented["mask"]

visualize(
    image_scaled, mask_scaled, original_image=image8, original_mask=mask8
)


# %%
# combine different transformations
aug = Compose([VerticalFlip(p=0.5), RandomRotate90(p=0.5)])

augmented = aug(image=image, mask=mask)

image_light = augmented["image"]
mask_light = augmented["mask"]

visualize(image_light, mask_light, original_image=image, original_mask=mask)
# %%

# Medium augmentations
aug = Compose(
    [
        OneOf(
            [
                RandomSizedCrop(
                    min_max_height=(50, 101),
                    height=original_height,
                    width=original_width,
                    p=0.5,
                ),
                PadIfNeeded(
                    min_height=original_height, min_width=original_width, p=0.5
                ),
            ],
            p=1,
        ),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        OneOf(
            [
                ElasticTransform(
                    p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                ),
                GridDistortion(p=0.5),
                OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5),
            ],
            p=0.8,
        ),
    ]
)

augmented = aug(image=image, mask=mask)

image_medium = augmented["image"]
mask_medium = augmented["mask"]

visualize(image_medium, mask_medium, original_image=image, original_mask=mask)
# %%

# Non-spatial stransformations
aug = Compose(
    [
        OneOf(
            [
                RandomSizedCrop(
                    min_max_height=(50, 90),
                    height=original_height,
                    width=original_width,
                    p=0.5,
                ),
                PadIfNeeded(
                    min_height=original_height, min_width=original_width, p=0.5
                ),
            ],
            p=1,
        ),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        OneOf(
            [
                ElasticTransform(
                    p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                ),
                GridDistortion(p=0.5),
                OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5),
            ],
            p=0.8,
        ),
        # CLAHE(p=0.8), # ONLY SUPORTS UINT8
        RandomBrightnessContrast(p=0.8),
        RandomGamma(p=0.8),
    ]
)

augmented = aug(image=image, mask=mask)

image_heavy = augmented["image"]
mask_heavy = augmented["mask"]

visualize(image_heavy, mask_heavy, original_image=image, original_mask=mask)

# %%
