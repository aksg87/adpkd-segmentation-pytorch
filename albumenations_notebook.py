# Notebook to check different augmentations
# TODO: use the new dataset formulation in `new_datasets.datasets`

# %%

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from data_set.datasets import SegmentationDataset
from data.data_utils import make_dcmdicts, get_labeled
from data.link_data import makelinks
from model.model_utils import get_preprocessing, preprocessing_fn

from data_set.transforms import T_x, T_y

# https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example_kaggle_salt.ipynb
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    RandomResizedCrop,
    OneOf,
    # CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate
)

# %%
IMG_IDX = (
    200
)  # SET THIS INDEX for selecting img label in augmentations example
# %%
makelinks()
# %%
# find number of patients
_, patient2dcm = make_dcmdicts(tuple(get_labeled()))
all_IDS = range(len(patient2dcm))
# %%
train_IDS, test_IDS = train_test_split(all_IDS, test_size=0.15, random_state=1)
train_IDS, val_IDS = train_test_split(
    train_IDS, test_size=0.176, random_state=1
)

# %%
dataset_train = SegmentationDataset(
    patient_IDS=train_IDS,
    transform_x=T_x,
    transform_y=T_y,
    preprocessing=get_preprocessing(preprocessing_fn),
)


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
def show_img(img, figsize=(8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img)
    plt.imshow(img, cmap="gray")


# %%
image, mask = dataset_train[IMG_IDX]
mask = mask[0] + mask[1]  # Combine into channel
image = image[0]  # Display only one channel
# %%
show_img(image)
# %%
# ORIGINAL
visualize(image, mask)

# %%
# PADDING
aug = PadIfNeeded(p=1, min_height=128, min_width=128)

augmented = aug(image=image, mask=mask)

image_padded = augmented["image"]
mask_padded = augmented["mask"]

print(image_padded.shape, mask_padded.shape)

visualize(image_padded, mask_padded, original_image=image, original_mask=mask)

# %%

# CENTER CROP
original_height, original_width = 96, 96

aug = CenterCrop(p=1, height=original_height, width=original_width)

augmented = aug(image=image_padded, mask=mask_padded)

image_center_cropped = augmented["image"]
mask_center_cropped = augmented["mask"]

print(image_center_cropped.shape, mask_center_cropped.shape)

assert (image - image_center_cropped).sum() == 0
assert (mask - mask_center_cropped).sum() == 0

visualize(
    image_padded,
    mask_padded,
    original_image=image_center_cropped,
    original_mask=mask_center_cropped,
)

# %%

# CROP
original_height, original_width = 96, 96

x_min = (128 - original_width) // 2
y_min = (128 - original_height) // 2

x_max = x_min + original_width
y_max = y_min + original_height

aug = Crop(p=1, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

augmented = aug(image=image_padded, mask=mask_padded)

image_cropped = augmented["image"]
mask_cropped = augmented["mask"]

print(image_cropped.shape, mask_cropped.shape)

assert (image - image_cropped).sum() == 0
assert (mask - mask_cropped).sum() == 0

visualize(
    image_cropped,
    mask_cropped,
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
aug = GridDistortion(p=1)

augmented = aug(image=image, mask=mask)

image_grid = augmented["image"]
mask_grid = augmented["mask"]

visualize(image_grid, mask_grid, original_image=image, original_mask=mask)

# %%
# Optical Distortion
aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)

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
aug = ShiftScaleRotate()

augmented = aug(image=image, mask=mask)

image_optical = augmented["image"]
mask_optical = augmented["mask"]

visualize(
    image_optical, mask_optical, original_image=image, original_mask=mask
)

# %%

# RandomSizedCrop

aug = RandomSizedCrop(
    p=1, min_max_height=(50, 96), height=72, width=72
)

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
        # CLAHE(p=0.8), # ONLY SUPORTS UINT8 # TODO: Convert input to Uint8 and check?
        RandomBrightnessContrast(p=0.8),
        RandomGamma(p=0.8),
    ]
)

augmented = aug(image=image, mask=mask)

image_heavy = augmented["image"]
mask_heavy = augmented["mask"]

visualize(image_heavy, mask_heavy, original_image=image, original_mask=mask)
