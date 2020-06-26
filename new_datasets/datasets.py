import numpy as np
import torch

import data_set.datasets as ds

from data.data_utils import (
    get_labeled,
    make_dcmdicts,
    new_path_2dcm,
    path_2label,
    get_y_Path,
)


class NewSegmentationDataset(torch.utils.data.Dataset):
    """Some information about NewSegmentationDataset"""

    def __init__(
        self,
        label2mask,
        dcm2attribs,
        patient2dcm,
        patient_indices=None,
        augmentation=None,
        smp_preprocessing=None,
    ):

        super().__init__()
        self.label2mask = label2mask
        self.patient_indices = patient_indices
        self.augmentation = augmentation
        self.smp_preprocessing = smp_preprocessing

        self.dcm2attribs = dcm2attribs
        self.pt2dcm = patient2dcm
        self.patients = list(patient2dcm.keys())

        # select subset of data (train, val, or test)
        if patient_indices is not None:
            self.patients = [self.patients[i] for i in patient_indices]

        self.dcm_paths = []
        for p in self.patients:
            self.dcm_paths.extend(patient2dcm[p])
        self.label_paths = [get_y_Path(dcm) for dcm in self.dcm_paths]

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [self[ii] for ii in range(*index.indices(len(self)))]

        # numpy uint8, (H, W)
        image = new_path_2dcm(self.dcm_paths[index])
        label = path_2label(self.label_paths[index])

        # numpy uint8, one hot encoded (C, H, W)
        mask = self.label2mask(label[np.newaxis, ...])

        if self.augmentation is not None:
            # requires (H, W, C) or (H, W)
            mask = mask.transpose(1, 2, 0)
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
            # get back to (C, H, W)
            mask = mask.transpose(2, 0, 1)

        # convert to float
        image = (image / 255).astype(np.float32)
        mask = mask.astype(np.float32)

        # smp preprocessing requires (H, W, 3)
        if self.smp_preprocessing is not None:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)
            image = self.smp_preprocessing(image).astype(np.float32)
            # get back to (3, H, W)
            image = image.transpose(2, 0, 1)
        else:
            # stack image to (3, H, W)
            image = np.repeat(image[np.newaxis, ...], 3, axis=0)

        return image, mask

    def __len__(self):
        return len(self.dcm_paths)

    def get_verbose(self, index):

        sample = self[index]
        dcm_path = self.dcm_paths[index]
        attribs = self.dcm2attribs[dcm_path]

        return sample, dcm_path, attribs


class NewBaselineDatasetGetter:
    """Create baseline segmentation dataset"""

    def __init__(
        self,
        splitter,
        splitter_key,
        label2mask,
        augmentation=None,
        smp_preprocessing=None,
        filters=None,
    ):
        super().__init__()
        self.splitter = splitter
        self.splitter_key = splitter_key
        self.label2mask = label2mask
        self.augmentation = augmentation
        self.smp_preprocessing = smp_preprocessing
        self.filters = filters

        dcms_paths = sorted(get_labeled())
        print(
            "The number of images before splitting and filtering: {}".format(
                len(dcms_paths)
            )
        )
        self.dcm2attribs, self.patient2dcm = make_dcmdicts(tuple(dcms_paths))
        if filters is not None:
            self.dcm2attribs, self.patient2dcm = filters(
                self.dcm2attribs, self.patient2dcm
            )
        self.all_idxs = range(len(self.patient2dcm.keys()))
        self.patient_indices = self.splitter(self.all_idxs)[self.splitter_key]

    def __call__(self):

        return NewSegmentationDataset(
            label2mask=self.label2mask,
            dcm2attribs=self.dcm2attribs,
            patient2dcm=self.patient2dcm,
            patient_indices=self.patient_indices,
            augmentation=self.augmentation,
            smp_preprocessing=self.smp_preprocessing,
        )


class BaselineDatasetGetter:
    """Create baseline segmentation dataset"""

    def __init__(
        self,
        splitter,
        transform_x,
        transform_y,
        preprocess_func,
        filters,
        splitter_key,
    ):
        super().__init__()
        self.splitter = splitter()
        self.transform_x = transform_x()
        self.transform_y = transform_y()
        self.preprocess_func = preprocess_func()
        self.filters = filters
        self.splitter_key = splitter_key

    def __call__(self):

        return ds.SegmentationDataset(
            patient_IDS=self.splitter[self.splitter_key],
            transform_x=self.transform_x,
            transform_y=self.transform_y,
            preprocessing=self.preprocess_func,
            filters=self.filters,
        )
