# Variational inference for symbolic regression using Miejer G functions

from algorithms.algorithm import Algorithm
from utils.meijer_g import MeijerG
import math
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
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

        self._opt_params = config['opt_params']

        self._prior_mu = config['prior_mu']
        self._prior_sigma = config['prior_sigma']
        self._prior = self.create_prior(config)

        self._q_mu = config['init_q_mu']
        self._q_sigma = config['init_q_sigma']

        self._num_steps = config['num_steps']
        self._learning_rate = config['learning_rate']

        self._num_samples = config['num_samples']

        self._max_abs_grad = config.get('max_abs_grad', None)

    def train(self, data):

        self.train_point_estimate(data)
        # self.train_mu(data)
        # self.train_maximum_likelihood(data)
        # self.train_variational_inference(data)

    def train_variational_inference(self, data):

        for i in range(self._num_steps):

            print('Step:', i)

            # Sample thetas
            thetas = self.sample_thetas()

            elbos = []
            elbo_grads = []

            for theta in thetas:

                # Set theta values
                theta_dict = self.get_theta_dict(theta)
                f = MeijerG(theta=self.get_theta(params=theta_dict),
                            order=self._order)
                # print(f.expression())

                # Calculate log likelihood p(x|z)
                log_likelihood = 0.0
                for x, y in zip(data['x'], data['y']):
                    log_likelihood += norm.logpdf(y, loc=f.evaluate(x))
                log_likelihood /= len(data['y'])

                # Calculate p(z) (gaussian)
                prior = self._prior.pdf(theta)
                log_prior = math.log(prior)

                # Calculate q(z)
                q_z = multivariate_normal.pdf(theta,
                                              mean=self._q_mu,
                                              cov=self._q_sigma)
                log_q_z = math.log(q_z)

                # Calculate ELBO
                elbo = log_likelihood + log_prior - log_q_z
                elbos.append(elbo)

                # Calculate log likelihood grad
                ll_grads = np.array([(y - f.evaluate(x))
                                     * f.gradients(x)[self.get_theta_idxs()]
                                     for x, y in zip(data['x'], data['y'])])
                ll_grads = np.mean(ll_grads, axis=0)
                # ll_grads = np.sum(ll_grads, axis=0)

                # Calculate log prior grad
                lp_grads = self._prior_mu - theta

                # Calculate log q(z) grad
                lq_grad = 0

                # Calculate ELBO grad
                elbo_grad = ll_grads + lp_grads - lq_grad

                # Perform gradient clipping
                elbo_grad = self.gradient_clipping(elbo_grad)

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

            elbos = np.array(elbos).ravel()
            elbo_grads = np.vstack(elbo_grads)

            # Calculate average ELBO and ELBO gradients
            avg_elbo = np.mean(np.array(elbos))
            avg_elbo_grad = np.mean(np.array(elbo_grads), axis=0)

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

        # TODO: Remove
        # theta[1], [2] and [4] do not produce hypergeometric functions
        # theta = [2, 2, 2, 1, 1]
        # order = [0, 1, 3, 1]

        # theta = [2]
        # order = [1, 0, 0, 2]

        # theta = [1, 1, 1]
        # theta = [1]
        # order = [2, 1, 2, 3]

        # theta = [1, 1, 1, 1, 1, 1]
        # order = [2, 2, 3, 3]

        # theta = [1, 1, 1, 1]
        # order = [2, 0, 1, 3]

        # theta = [1, 1, 1, 0, 1]
        # order = [1, 2, 2, 2]

        theta = [-2, 0, 1]
        order = [1, 1, 1, 2]

        for i in range(self._num_steps):

            f = MeijerG(theta=theta, order=order)

            print('Theta:', theta)
            print(f.expression())

            # Calculate loss
            losses = []
            for x, y in zip(data['x'], data['y']):
                loss = math.pow(f.evaluate(x) - y, 2)
                losses.append(loss)
            loss = np.sum(np.array(losses))

            print('Loss:', loss)

            # Calculate loss grads w.r.t theta
            loss_grads = []
            for x, y in zip(data['x'], data['y']):
                loss_grad = 2 * (f.evaluate(x) - y) * f.gradients(x)
                loss_grads.append(loss_grad)
            loss_grads = np.sum(np.array(loss_grads), axis=0)

            # Clip gradients
            loss_grads = np.clip(loss_grads, -10, 10)
            print('Grads:', loss_grads)

            # Take optimisation step
            theta[0] -= self._learning_rate * loss_grads[0]
            # theta[1] -= self._learning_rate * loss_grads[1]
            # theta[2] -= self._learning_rate * loss_grads[2]
            # theta[3] -= self._learning_rate * loss_grads[3]
            # theta[4] -= self._learning_rate * loss_grads[4]

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

    def create_prior(self, config):
        return multivariate_normal(config['prior_mu'], config['prior_sigma'])

    # Sample thetas from q
    def sample_thetas(self):

        # Sample from q
        thetas = multivariate_normal.rvs(mean=self._q_mu, cov=self._q_sigma,
                                         size=self._num_samples)

        # Reshape to correct numpy array size
        thetas = np.reshape(thetas, (self._num_samples, len(self._opt_params)))

        return thetas

    # Convert numpy array of thetas to a dictionary where the theta values
    # are labelled by their parameter names
    def get_theta_dict(self, thetas):

        assert len(thetas) == len(self._opt_params)

        theta_dict = {}

        for param, theta in zip(self._opt_params, thetas):
            theta_dict[param] = theta

        return theta_dict

    # Get theta indices according to parameters being optimised
    def get_theta_idxs(self):

        idxs = []

        for p in self._opt_params:

            if p == 'c':
                idxs.append(len(self._a) + len(self._b))

            elif p.startswith('a'):
                idxs.append(int(p[1]) - 1)

            elif p.startswith('b'):
                idxs.append(int(p[1]) + len(self._a) - 1)

        idxs.sort()
        return idxs

    # Peform gradient clipping on the gradients
    def gradient_clipping(self, grads):

        if self._max_abs_grad is not None:
            for i, g in enumerate(grads):
                if abs(g) > self._max_abs_grad:
                    grads[i] = self._max_abs_grad * np.sign(g)

        return grads
