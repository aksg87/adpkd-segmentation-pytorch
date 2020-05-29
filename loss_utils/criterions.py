"""Dictionary of criterions"""


class LossesMetrics:
    """ Produces function to generate dict of keys: losses/metrics for batch"""

    def __init__(self, criterions_dict):
        """
        Arguments:
            criterions_dict {dict} -- key (str) : criterion_losses
        """
        self.criterions_dict = criterions_dict

    def __call__(self):
        def losses_dict(y, y_hat):
            res = {}

            for key, criterion in self.criterions_dict.items():
                res[key] = criterion(y, y_hat)

            return res

        return losses_dict
