"""Dictionary of criterions"""

from config.config_utils import get_object_instance


class LossesMetrics:
    """ Produces function to generate dict of keys: losses/metrics for batch"""

    def __init__(self, criterions_dict):
        """
        Arguments:
            criterions_dict {dict} -- key (str) : loss_configs (yaml config)
        """

        self.criterions_dict = {}

        # generates object for loss configs
        for key, value in criterions_dict.items():
            self.criterions_dict[key] = get_object_instance(value)

    def __call__(self):
        def losses_dict(y, y_hat):
            res = {}

            for key, criterion in self.criterions_dict.items():
                res[key] = criterion(y, y_hat)

            return res

        return losses_dict
