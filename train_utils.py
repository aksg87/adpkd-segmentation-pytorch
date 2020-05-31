"""Train utils"""

from torch import optim


class TorchLRScheduler():
    def __init__(self, name, param_dict):
        self.param_dict = param_dict
        self.scheduler = getattr(optim.lr_scheduler, name)

    def __call__(self, optimizer):
        return self.scheduler(optimizer, **self.param_dict)
