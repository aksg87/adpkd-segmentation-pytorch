"""Train utils"""

from torch import optim


class TorchLRScheduler():
    def __init__(self, name, param_dict, step_type):
        self.param_dict = param_dict
        self.scheduler = getattr(optim.lr_scheduler, name)
        self.step_type = step_type

    def __call__(self, optimizer):
        return self.scheduler(optimizer, **self.param_dict)
