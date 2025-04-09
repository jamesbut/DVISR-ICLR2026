from domains.domain import Domain
import numpy as np


class Linear(Domain):

    def __init__(self, config):

        super().__init__(config)

        self._m = config['m']
        self._c = config['c']

    def evaluate(self, x):

        assert x.shape[1] == 1

        y = self._m * x + self._c

        return np.ravel(y)

    def create_x(self, config, num_vals=None):
        return Domain.evenly_spaced_x(config, num_vals)
