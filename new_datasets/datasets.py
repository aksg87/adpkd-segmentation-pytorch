# %%

import data_set.datasets as ds

# %%


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
