from domains.domain import Domain
import numpy as np


class Quadratic(Domain):

    def __init__(self, config):

        super().__init__(config)

        self._a = config['a']
        self._b = config['b']
        self._c = config['c']

    def evaluate(self, x):

        assert x.shape[1] == 1

        y = self._a * x * x + self._b * x + self._c

        return np.ravel(y)
