""""Utility functions for dataset filtering"""
from abc import ABC, abstractmethod
from collections import OrderedDict

from adpkd_segmentation.data.data_utils import PATIENT, SEQUENCE


class Filtering(ABC):
    @abstractmethod
    def criterion(self, attribs):
        pass

    def __call__(self, dcm2attribs, patient2dcm):
        new_dcm2attribs = OrderedDict()
        new_patient2dcm = OrderedDict()

        for dcm, attribs in dcm2attribs.items():
            if self.criterion(attribs):
                new_dcm2attribs[dcm] = attribs
                new_patient2dcm.setdefault(attribs[PATIENT], []).append(
                    dcm
                )

        return new_dcm2attribs, new_patient2dcm


class SequenceFiltering(Filtering):
    def __init__(self, sequence_list):
        self.sequence_list = sequence_list

    def criterion(self, attribs):
        return attribs[SEQUENCE] in self.sequence_list


class PatientFiltering(Filtering):
    def __init__(self, patient_IDS):
        self.patient_IDS = patient_IDS

    def criterion(self, attribs):
        return attribs[PATIENT] in self.patient_IDS
