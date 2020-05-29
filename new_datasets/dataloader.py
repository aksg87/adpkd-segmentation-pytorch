from torch.utils.data import DataLoader


class BaselineDataloaderGetter:
    """create baseline dataloaders"""

    def __init__(self, datasets, batchsize, shuffle):
        """
        Arguments:
            datasets {list of dataset}
            batchsize {int}
            shuffle {boolean}
        """
        self.datasets = datasets()
        self.batchsize = batchsize
        self.shuffle = shuffle

    def __call__(self):

        return [
            DataLoader(
                dataset=dataset,
                batch_size=self.batchsize,
                shuffle=self.shuffle,
            )
            for dataset in self.datasets
        ]  # convert to dictionary with names in datasets?
