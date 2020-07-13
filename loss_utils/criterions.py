"""Dictionary of criterions"""


class LossesMetrics:
    """ Produces function to generate dict of keys: losses/metrics for batch"""

    def __init__(self, criterions_dict, requires_extra_info=None):
        """
        Args:
            criterions_dict {dict} -- key (str) : criterion_losses
            requires_extra_info {dict}, mapping keys in `criterions_dict`
                to True or False
            (tells if extra info will be used for losses)
        """
        self.criterions_dict = criterions_dict
        self.requires_extra_info = requires_extra_info
        if requires_extra_info is None:
            self.requires_extra_info = {
                c_name: False for c_name in criterions_dict
            }

    def __call__(self):
        def losses_dict(y_hat, y, extra_dict=None):
            res = {}
            for c_name, criterion in self.criterions_dict.items():
                # optional info for some criterions
                if self.requires_extra_info[c_name]:
                    res[c_name] = criterion(y_hat, y, extra_dict)
                else:
                    res[c_name] = criterion(y_hat, y)
            return res

        return losses_dict
