# Variational inference for symbolic regression using Miejer G functions

from algorithms.algorithm import Algorithm
from utils.meijer_g import MeijerG
import math
import numpy as np
from scipy.stats import norm
np.set_printoptions(suppress=True)


class VIMG(Algorithm):

    def __init__(self, config):

        self._theta = [2, 2, 2, 1, 1]
        self._order = [0, 1, 3, 1]

        self._num_steps = config['num_steps']
        self._learning_rate = config['learning_rate']

        self._mu = 0.0
        self._sigma = 1.0

    def train(self, data):

        #self.train_point_estimate(data)

        #self.train_mu(data)

        self.train_maximum_likelihood(data)

    #def train_maximum_likelihood(self, data):

    def train_mu(self, data):

        for i in range(self._num_steps):

            # Sample theta
            theta = self._mu + norm.rvs()
            self._theta[4] = theta

            f = MeijerG(theta=self._theta, order=self._order)

            # Calculate loss
            loss = self.calculate_loss(f, data)

            # Calculate loss gradient w.r.t. theta
            loss_grads = self.calculate_grads(f, data)

            print(f.expression())
            print('θ_4:', self._theta[4])
            print('θ_4 grad:', loss_grads[4])
            print('mu:', self._mu)
            print('Loss:', loss)
            print('--------------')

            # Take optimisation step
            self._mu -= self._learning_rate * loss_grads[4]

    def train_point_estimate(self, data):

        for i in range(self._num_steps):

            f = MeijerG(theta=self._theta, order=self._order)

            # Calculate loss
            loss = self.calculate_loss(f, data)

            # Calculate loss gradient w.r.t. theta
            loss_grads = self.calculate_grads(f, data)

            print(f.expression())
            print('θ:', self._theta)
            print('Loss:', loss)

            # Take optimisation step
            self.optimise(loss_grads)

    def calculate_loss(self, f, data):

        loss = np.array([math.pow(f.evaluate(x) - y, 2)
                         for x, y in zip(data['x'], data['y'])])
        loss = np.sum(loss)

        return loss

    def calculate_grads(self, f, data):

        # Gradients take a while to compute
        loss_grads = np.array([2 * (f.evaluate(x) - y) * np.concatenate(f.gradients(x))
                              for x, y in zip(data['x'], data['y'])])
        loss_grads = np.sum(loss_grads, axis=0)

        return loss_grads

    def optimise(self, loss_grads):
        self._theta[4] -= self._learning_rate * loss_grads[4]
