# %%

import data_set.datasets as ds

# %%
class BaselineDatasetGetter:
    """Create baseline segmentation datasets"""

    def __init__(self, splitter, transform_x, transform_y, preprocess_func):
        super().__init__()
        self.splitter = splitter()
        self.transform_x = transform_x()
        self.transform_y = transform_y()
        self.preprocess_func = preprocess_func()

    def __call__(self):

        return [
            ds.SegmentationDataset(
                patient_IDS=indices,
                transform_x=self.transform_x,
                transform_y=self.transform_y,
                preprocessing=self.preprocess_func,
            )
            for indices in self.splitter
        ]
