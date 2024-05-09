# Variational inference for symbolic regression using Miejer G functions

from algorithms.algorithm import Algorithm
from utils.meijer_g import MeijerG
import math
import numpy as np
np.set_printoptions(suppress=True)


class VIMG(Algorithm):

    def __init__(self, config):

        self._theta = [2, 2, 2, 1, 1]
        self._order = [0, 1, 3, 1]

        self._num_steps = config['num_steps']
        self._learning_rate = config['learning_rate']

    def train(self, data):

        for i in range(self._num_steps):

            f = MeijerG(theta=self._theta, order=self._order)
            print(f.expression())

            # Calculate loss
            loss = np.array([math.pow(f.evaluate(x) - y, 2)
                             for x, y in zip(data['x'], data['y'])])
            loss = np.sum(loss)

            print('θ:', self._theta)
            print('Loss:', loss)

            # Calculate loss gradient w.r.t. theta
            # Gradients take a while to compute
            loss_grads = np.array([2 * (f.evaluate(x) - y) * np.concatenate(f.gradients(x))
                                  for x, y in zip(data['x'], data['y'])])
            loss_grads = np.sum(loss_grads, axis=0)

            # Take optimisation step
            self._theta[4] -= self._learning_rate * loss_grads[4]
