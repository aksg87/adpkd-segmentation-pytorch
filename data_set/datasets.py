#%%
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from data.data_utils import get_dcms_paths, get_labeled, make_dcmdicts, path_2dcm, path_2label
from data.link_data import makelinks
# %%
class SegmentationDataset(torch.utils.data.Dataset):
    """Some Information about SegmentationDataset"""
    def __init__(self, patient_IDS=None, hyperparams=None, transform_x=None, transform_y=None, preprocessing=None):

        super(SegmentationDataset, self).__init__()
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.preprocessing = preprocessing # --> for segmentation-models-pytorch
        makelinks()
        
        dcms_paths = get_labeled()

        dcm2attribs, patient2dcm = make_dcmdicts(tuple(dcms_paths))

        self.dcm2attribs = dcm2attribs
        self.pt2dcm = patient2dcm
        self.patients = list(patient2dcm.keys())

        # select subset of data for train, val, test split
        if patient_IDS:
            self.patients = [self.patients[i] for i in patient_IDS]
        
        patient_dcms = []
        for p in self.patients:
            patient_dcms.extend(patient2dcm[p])
        
        self.dcm_paths = sorted(patient_dcms)
        self.label_paths = [get_y_Path(dcm) for dcm in self.dcm_paths]

    def __getitem__(self, index):
        
        if isinstance(index, slice):
            return [self[ii] for ii in range(*index.indices(len(self)))]

        dcm = path_2dcm(self.dcm_paths[index])
        label = path_2label(self.label_paths[index])

        dcm = self.transform_x(dcm)
        label = self.transform_y(label)

        if self.preprocessing: # required for segmentation-models-pytorch --> essentially normalizes data for encoder, and transforms label to float32 dtype
            sample = self.preprocessing(image=dcm, mask=label)
            dcm, label = sample['image'], sample['mask'] 

        return dcm, label

    def __len__(self):
        return len(self.dcm_paths)

    def get_verbose(self, index):

        sample = self[index]
        dcm_path = self.dcm_paths[index]
        attribs = self.dcm2attribs[dcm_path]

        return sample, dcm_path, attribs
