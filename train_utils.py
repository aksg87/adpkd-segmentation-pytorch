"""Train utils"""

import torch
from torch import optim


class TorchLRScheduler:
    def __init__(self, name, param_dict, step_type):
        self.param_dict = param_dict
        self.scheduler = getattr(optim.lr_scheduler, name)
        self.step_type = step_type

    def __call__(self, optimizer):
        return self.scheduler(optimizer, **self.param_dict)


def save_model_data(path, model, global_step):
    torch.save(
        {"global_step": global_step, "model_state_dict": model.state_dict()},
        path,
    )


def load_model_data(path, model, new_format=False):
    checkpoint = torch.load(path)
    global_step = 0
    if not new_format:
        model.load_state_dict(checkpoint)
    else:
        global_step = checkpoint["global_step"]
        model.load_state_dict(checkpoint["model_state_dict"])
    return global_step
