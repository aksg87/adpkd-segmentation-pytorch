""""Utility functions for dataset filtering"""
from abc import ABC, abstractmethod
from collections import OrderedDict

PATIENT_KEY = "patient"
SEQUENCE_KEY = "seq"


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
                new_patient2dcm.setdefault(attribs[PATIENT_KEY], []).append(
                    dcm
                )

        return new_dcm2attribs, new_patient2dcm


class SequenceFiltering(Filtering):
    def __init__(self, sequence_list):
        self.sequence_list = sequence_list

    def criterion(self, attribs):
        return attribs[SEQUENCE_KEY] in self.sequence_list
