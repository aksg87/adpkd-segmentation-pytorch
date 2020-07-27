from torch.utils.data import DataLoader


class BaselineDataloaderGetter:
    """create baseline dataloader"""

    def __init__(self, dataset, batchsize, shuffle):
        """
        Args:
            dataset: `SegmentationDataset`
            batchsize: int, batch size
            shuffle: bool , whether to shuffle after each epoch
        """
        self.dataset = dataset()
        self.batchsize = batchsize
        self.shuffle = shuffle

    def __call__(self):
        return DataLoader(
                dataset=self.dataset,
                batch_size=self.batchsize,
                shuffle=self.shuffle,
            )
