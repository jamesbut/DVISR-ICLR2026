# Variational inference for symbolic regression using Miejer G functions

from algorithms.algorithm import Algorithm
from utils.meijer_g import MeijerG
import math
import numpy as np
from scipy.stats import norm
np.set_printoptions(suppress=True)


class VIMG(Algorithm):

    def __init__(self, config):

        self._m = config['m']
        self._n = config['n']
        self._p = config['p']
        self._q = config['q']

        self._order = [self._m, self._n, self._p, self._q]

        self._a = config['a']
        self._b = config['b']
        self._c = config['c']

        self.get_theta()

        self._num_steps = config['num_steps']
        self._learning_rate = config['learning_rate']

        self._q_mu = config['init_q_mu']
        self._q_sigma = config['init_q_sigma']

        self._num_samples = config['num_samples']

        self._prior_mu = config['prior_mu']
        self._prior_sigma = config['prior_sigma']

    def train(self, data):

        # self.train_point_estimate(data)
        # self.train_mu(data)
        # self.train_maximum_likelihood(data)
        self.train_variational_inference(data)

    def train_variational_inference(self, data):

        for i in range(self._num_steps):

            print('Step:', i)

            # Sample thetas
            thetas = [norm.rvs() + self._q_mu for i in range(self._num_samples)]

            elbos = []
            elbo_grads = []

            for theta in thetas:

                # Set theta[4]
                th = self._theta.copy()
                th[4] = theta
                f = MeijerG(theta=th, order=self._order)

                # Calculate log likelihood p(x|z)
                log_likelihood = 0.0
                for x, y in zip(data['x'], data['y']):
                    log_likelihood += norm.logpdf(y, loc=f.evaluate(x))
                #log_likelihood /= len(data['y'])

                # Calculate p(z) (gaussian)
                prior = norm.pdf((theta - self._prior_mu) / self._prior_sigma)
                log_prior = math.log(prior)

                # Calculate q(z)
                q_z = norm.pdf((theta - self._q_mu) / self._q_sigma)
                log_q_z = math.log(q_z)

                # Calculate ELBOs
                elbo = log_likelihood + log_prior - log_q_z
                elbos.append(elbo)

                # Calculate log likelihood grad
                ll_grads = np.array([(y - f.evaluate(x)) * f.gradients(x)[4]
                                     for x, y in zip(data['x'], data['y'])])
                #ll_grad = np.mean(ll_grads)
                ll_grad = np.sum(ll_grads)

                # Calculate log prior grad
                lp_grad = self._prior_mu - theta

                # Calculate log q(z) grad
                lq_grad = 0

                elbo_grad = ll_grad + lp_grad - lq_grad
                elbo_grads.append(elbo_grad)

                '''
                print(f.expression())

                print('log likelihood:', log_likelihood)
                print('log prior:', log_prior)
                print('log_q_z:', log_q_z)
                print('ELBO:', elbo)

                print('log likelihood grad:', ll_grad)
                print('log prior grad:', lp_grad)
                print('log_q_z grad:', lq_grad)
                print('ELBO grad:', elbo_grad)
                '''

            # Calculate average ELBO and ELBO gradients
            avg_elbo = np.mean(np.array(elbos))
            avg_elbo_grad = np.mean(np.array(elbo_grads))

            # Calculate loss and loss gradients
            loss = -avg_elbo
            loss_grad = -avg_elbo_grad

            print('q_mu:', self._q_mu)
            print('q_mu grad:', loss_grad)
            print('Loss:', loss)
            print('--------------')

            # Take optimisation step
            self._q_mu -= self._learning_rate * loss_grad

    def train_maximum_likelihood(self, data):

        for i in range(self._num_steps):

            f = MeijerG(theta=self._theta, order=self._order)

            # Calculate log likelihood
            log_likelihood = 0.0
            for x, y in zip(data['x'], data['y']):
                log_likelihood += norm.logpdf(y, loc=f.evaluate(x))

            # Calculate loss
            loss = -log_likelihood

            # Calculate loss gradient w.r.t. mu
            loss_grads = np.array([(f.evaluate(x) - y) * f.gradients(x)[4]
                                   for x, y in zip(data['x'], data['y'])])
            loss_grad = np.sum(loss_grads)

            print(f.expression())
            print('θ_4:', self._theta[4])
            print('θ_4 grad:', loss_grad)
            print('Loss:', loss)
            print('--------------')

            # Take optimisation step
            self._theta[4] -= self._learning_rate * loss_grad

    def train_mu(self, data):

        for i in range(self._num_steps):

            # Sample theta
            theta = self._q_mu + norm.rvs()
            self._theta[4] = theta

            f = MeijerG(theta=self._theta, order=self._order)

            # Calculate loss
            loss = self.calculate_loss(f, data)

            # Calculate loss gradient w.r.t. mu
            loss_grads = self.calculate_grads(f, data)

            print(f.expression())
            print('θ_4:', self._theta[4])
            print('θ_4 grad:', loss_grads[4])
            print('mu:', self._q_mu)
            print('Loss:', loss)
            print('--------------')

            # Take optimisation step
            self._q_mu -= self._learning_rate * loss_grads[4]

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

    def get_theta(self, params=None):

        a = self._a.copy()
        b = self._b.copy()
        c = self._c

        if params is not None:

            for key, value in params.items():

                if key == 'c':
                    c = value

                elif key.startswith('a'):
                    idx = int(key[1])
                    a[idx - 1] = value

                elif key.startswith('b'):
                    idx = int(key[1])
                    b[idx - 1] = value

                else:
                    raise RuntimeError('Unrecognised parameter name '
                                       'in set_theta')

        theta = a + b + [c]

        return theta
