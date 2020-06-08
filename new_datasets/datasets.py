import torch

import data_set.datasets as ds

from data.data_utils import (
    get_labeled,
    make_dcmdicts,
    new_path_2dcm,
    path_2label,
    get_y_Path,
    filter_dcm2attribs,
)


class NewSegmentationDataset(torch.utils.data.Dataset):
    """Some information about NewSegmentationDataset"""

    def __init__(
        self,
        label2mask,
        patient_indices=None,
        augmentation=None,
        preprocessing=None,
        filters=None,
    ):

        super().__init__()
        self.label2mask = label2mask
        self.patient_indices = patient_indices
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.filters = filters

        dcms_paths = get_labeled()

        dcm2attribs, patient2dcm = make_dcmdicts(tuple(dcms_paths))

        if filters is not None:
            dcm2attribs = filter_dcm2attribs(filters, dcm2attribs)

        self.dcm2attribs = dcm2attribs
        self.pt2dcm = patient2dcm
        self.patients = list(patient2dcm.keys())

        # select subset of data for train, val, test split
        if patient_indices is not None:
            self.patients = [self.patients[i] for i in patient_indices]

        patient_dcms = []
        for p in self.patients:
            patient_dcms.extend(patient2dcm[p])

        self.dcm_paths = sorted(patient_dcms)
        self.label_paths = [get_y_Path(dcm) for dcm in self.dcm_paths]

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [self[ii] for ii in range(*index.indices(len(self)))]

        # numpy uint8, (H, W)
        image = new_path_2dcm(self.dcm_paths[index])
        label = path_2label(self.label_paths[index])

        # numpy uint8, one hot encoded (C, H, W)
        mask = self.label2mask(label)

        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # convert to float and stack
        image = image / 255.0
        image = np.repeat(image, 3, axis=0)

        # smp preprocessing
        if self.preprocessing is not None:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.dcm_paths)

    def get_verbose(self, index):

        sample = self[index]
        dcm_path = self.dcm_paths[index]
        attribs = self.dcm2attribs[dcm_path]

        return sample, dcm_path, attribs

# WIP
class NewBaselineDatasetGetter:
    """Create baseline segmentation dataset"""

    def __init__(
        self,
        label2mask,
        patient_indices=None,
        augmentation=None,
        preprocessing=None,
        filters=None,
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
