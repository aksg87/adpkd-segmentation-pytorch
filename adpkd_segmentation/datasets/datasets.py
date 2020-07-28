import json
import numpy as np
import torch

from adpkd_segmentation.data.data_utils import (
    get_labeled,
    get_y_Path,
    int16_to_uint8,
    make_dcmdicts,
    path_2dcm_int16,
    path_2label,
    TKV_update,
)

from adpkd_segmentation.data.data_utils import (
    KIDNEY_PIXELS,
    STUDY_TKV,
    VOXEL_VOLUME,
)

from adpkd_segmentation.datasets.filters import PatientFiltering


class NewSegmentationDataset(torch.utils.data.Dataset):
    """Some information about NewSegmentationDataset"""

    def __init__(
        self,
        label2mask,
        dcm2attribs,
        patient2dcm,
        patient_IDS=None,
        augmentation=None,
        smp_preprocessing=None,
        normalization=None,
        output_idx=False,
        attrib_types=None,
    ):

        super().__init__()
        self.label2mask = label2mask
        self.dcm2attribs = dcm2attribs
        self.pt2dcm = patient2dcm
        self.patient_IDS = patient_IDS
        self.augmentation = augmentation
        self.smp_preprocessing = smp_preprocessing
        self.normalization = normalization
        self.output_idx = output_idx
        self.attrib_types = attrib_types

        # store some attributes as PyTorch tensors
        if self.attrib_types is None:
            self.attrib_types = {
                STUDY_TKV: "float32",
                KIDNEY_PIXELS: "float32",
                VOXEL_VOLUME: "float32",
            }

        self.patients = list(patient2dcm.keys())
        # kept for compatibility with previous experiments
        # following patient order in patient_IDS
        if patient_IDS is not None:
            self.patients = patient_IDS

        self.dcm_paths = []
        for p in self.patients:
            self.dcm_paths.extend(patient2dcm[p])
        self.label_paths = [get_y_Path(dcm) for dcm in self.dcm_paths]

        # study_id to TKV and TKV for each dcm
        self.studies, self.dcm2attribs = TKV_update(dcm2attribs)
        # storring attrib types as tensors
        self.tensor_dict = self.prepare_tensor_dict(self.attrib_types)

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [self[ii] for ii in range(*index.indices(len(self)))]

        # numpy int16, (H, W)
        im_path = self.dcm_paths[index]
        image = path_2dcm_int16(im_path)
        # image local scaling by default to convert to uint8
        if self.normalization is None:
            image = int16_to_uint8(image)
        else:
            image = self.normalization(image, self.dcm2attribs[im_path])

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

        if self.output_idx:
            return image, mask, index
        return image, mask

    def __len__(self):
        return len(self.dcm_paths)

    def get_verbose(self, index):

        sample = self[index]
        dcm_path = self.dcm_paths[index]
        attribs = self.dcm2attribs[dcm_path]

        return sample, dcm_path, attribs

    def get_extra_dict(self, batch_of_idx):
        return {k: v[batch_of_idx] for k, v in self.tensor_dict.items()}

    def prepare_tensor_dict(self, attrib_types):
        tensor_dict = {}
        for k, v in attrib_types.items():
            tensor_dict[k] = torch.zeros(
                self.__len__(), dtype=getattr(torch, v)
            )

        for idx, _ in enumerate(self):
            dcm_path = self.dcm_paths[idx]
            attribs = self.dcm2attribs[dcm_path]
            for k, v in tensor_dict.items():
                v[idx] = attribs[k]

        return tensor_dict


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
        normalization=None,
        output_idx=False,
        attrib_types=None,
    ):
        super().__init__()
        self.splitter = splitter
        self.splitter_key = splitter_key
        self.label2mask = label2mask
        self.augmentation = augmentation
        self.smp_preprocessing = smp_preprocessing
        self.filters = filters
        self.normalization = normalization
        self.output_idx = output_idx
        self.attrib_types = attrib_types

        dcms_paths = sorted(get_labeled())
        print(
            "The number of images before splitting and filtering: {}".format(
                len(dcms_paths)
            )
        )
        dcm2attribs, patient2dcm = make_dcmdicts(tuple(dcms_paths))

        if filters is not None:
            dcm2attribs, patient2dcm = filters(dcm2attribs, patient2dcm)

        self.all_patient_IDS = list(patient2dcm.keys())
        # train, val, or test
        self.patient_IDS = self.splitter(self.all_patient_IDS)[
            self.splitter_key
        ]

        patient_filter = PatientFiltering(self.patient_IDS)
        self.dcm2attribs, self.patient2dcm = patient_filter(
            dcm2attribs, patient2dcm
        )
        if self.normalization is not None:
            self.normalization.update_dcm2attribs(self.dcm2attribs)

    def __call__(self):
        return NewSegmentationDataset(
            label2mask=self.label2mask,
            dcm2attribs=self.dcm2attribs,
            patient2dcm=self.patient2dcm,
            patient_IDS=self.patient_IDS,
            augmentation=self.augmentation,
            smp_preprocessing=self.smp_preprocessing,
            normalization=self.normalization,
            output_idx=self.output_idx,
            attrib_types=self.attrib_types,
        )


class JsonDatasetGetter:
    """Get the dataset from a prepared patient ID split"""

    def __init__(
        self,
        json_path,
        splitter_key,
        label2mask,
        augmentation=None,
        smp_preprocessing=None,
        normalization=None,
        output_idx=False,
        attrib_types=None,
    ):
        super().__init__()

        self.label2mask = label2mask
        self.augmentation = augmentation
        self.smp_preprocessing = smp_preprocessing
        self.normalization = normalization
        self.output_idx = output_idx
        self.attrib_types = attrib_types

        dcms_paths = sorted(get_labeled())
        print(
            "The number of images before splitting and filtering: {}".format(
                len(dcms_paths)
            )
        )
        dcm2attribs, patient2dcm = make_dcmdicts(tuple(dcms_paths))

        print("Loading ", json_path)
        with open(json_path, "r") as f:
            dataset_split = json.load(f)
        self.patient_IDS = dataset_split[splitter_key]

        # filter info dicts to correpsond to patient_IDS
        patient_filter = PatientFiltering(self.patient_IDS)
        self.dcm2attribs, self.patient2dcm = patient_filter(
            dcm2attribs, patient2dcm
        )
        if self.normalization is not None:
            self.normalization.update_dcm2attribs(self.dcm2attribs)

    def __call__(self):
        return NewSegmentationDataset(
            label2mask=self.label2mask,
            dcm2attribs=self.dcm2attribs,
            patient2dcm=self.patient2dcm,
            patient_IDS=self.patient_IDS,
            augmentation=self.augmentation,
            smp_preprocessing=self.smp_preprocessing,
            normalization=self.normalization,
            output_idx=self.output_idx,
            attrib_types=self.attrib_types,
        )
