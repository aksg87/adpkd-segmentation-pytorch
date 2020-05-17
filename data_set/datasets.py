#%%
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from data import *
# %%
class SegmentationDataset(torch.utils.data.Dataset):
    """Some Information about SegmentationDataset"""
    def __init__(self, patient_IDS=None, hyperparams=None, transform=None):

        super(SegmentationDataset, self).__init__()

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

        return dcm, label

    def __len__(self):
        return len(self.dcm_paths)
