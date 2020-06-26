from sklearn.model_selection import train_test_split


class GenSplit:
    def __init__(self, train=0.7, val=0.15, test=0.15, seed=1):
        super().__init__()

        self.train = train
        self.val = val
        self.test = test
        self.seed = seed

    def __call__(self, all_idxs):

        # split train from validation-test
        train_idxs, test_val_idxs = train_test_split(
            all_idxs, test_size=(self.val + self.test), random_state=self.seed
        )

        # split validation from test
        val_idxs, test_idxs = train_test_split(
            test_val_idxs,
            test_size=(self.test / (self.test + self.val)),
            random_state=self.seed,
        )

        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs

        print(
            "The number of (filtered) train patients: {}".format(
                len(self.train_idxs)
            )
        )
        print(
            "The number of (filtered) validation patients: {}".format(
                len(self.val_idxs)
            )
        )
        print(
            "The number of (filtered) test patients: {}".format(len(self.test_idxs))
        )

        return {
            "train": self.train_idxs,
            "val": self.val_idxs,
            "test": self.test_idxs,
        }
