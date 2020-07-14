"""Dictionary of criterions"""


class LossesMetrics:
    """ Produces function to generate dict of keys: losses/metrics for batch"""

    def __init__(self, criterions_dict, requires_extra_info=None):
        """
        Args:
            criterions_dict {dict} -- key (str) : criterion_losses
            requires_extra_info {list or None}, keys in `criterions_dict`
                noting losses which require extra information
        """
        self.criterions_dict = criterions_dict
        if requires_extra_info is None:
            self.requires_extra_info = set()
        else:
            self.requires_extra_info = set(requires_extra_info)

    def __call__(self):
        def losses_dict(y_hat, y, extra_dict=None):
            res = {}
            for c_name, criterion in self.criterions_dict.items():
                # optional info for some criterions
                if c_name in self.requires_extra_info:
                    res[c_name] = criterion(y_hat, y, extra_dict)
                else:
                    res[c_name] = criterion(y_hat, y)
            return res

        return losses_dict
