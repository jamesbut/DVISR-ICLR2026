# Variational Inference Symbolic Regression using categorical variables
# over tokens.

# This algorithm is similar to Deep Symbolic Regression in that it uses a
# sequence of categorical variables over a set of tokens to build an
# analytic equation.

from algorithms.algorithm import Algorithm
import numpy as np
import scipy
import math
import itertools
import matplotlib.pyplot as plt
import torch
from .q import q
from .equation import Equation, optimise_eq_consts
from .behaviour_policy import BehaviourPolicy
import copy
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd


class VICatSR(Algorithm, BaseEstimator, RegressorMixin):

    def __init__(self, config):

        # Prepare binary and unary operations and constants as tokens
        self._token_set = []
        self._token_id = 0

        if 'binary_ops' in config['operators']:
            for bo in config['operators']['binary_ops']:
                self._token_set.append({"op": bo, "type": "bin_op",
                                        "sub_type": None,
                                        "id": self._token_id})
                self._token_id += 1

        if 'unary_ops' in config['operators']:
            for uo in config['operators']['unary_ops']:
                self._token_set.append({"op": uo, "type": "un_op",
                                        "sub_type": None,
                                        "id": self._token_id})
                self._token_id += 1

        # Add constants as tokens
        if 'consts' in config['operators']:

            for c in config['operators']['consts']:

                token = {"op": c, "type": "const",
                         "sub_type": "float_const",
                         "id": self._token_id}
                if c == 'opt_const':
                    token['value'] = None

                self._token_set.append(token)
                self._token_id += 1

            self._distr_over_consts = False
            self._q_const_variance = None

        else:

            if 'distr_over_consts' in config and not config['distr_over_consts']:
                self._distr_over_consts = False
                self._q_const_variance = None

            else:

                self._token_set.append({"op": "distr_const", "type": "const",
                                        "sub_type": "float_const",
                                        "value": None,
                                        "id": self._token_id})
                self._token_id += 1
                self._distr_over_consts = True
                self._q_const_variance = config.get('q_const_variance', None)

        # Number of equations sampled to calculate expected loss
        self._num_eq_samples = config['num_eq_samples']

        # Maximum equation tree depth
        self._max_depth = None
        if 'max_depth' in config:
            raise NotImplementedError('Max tree depth not yet implemented')

        # Maximum number of tokens in generated equations
        self._max_num_tokens = config['max_num_tokens']

        # Learning rate for optimiser
        self._lr = config['target_policy']['learning_rate']

        # Size of RNN hidden layer
        self._hidden_layer_size = \
            config['target_policy']['rnn_hidden_layer_size']

        # Initialise GRU weights to 0
        self._init_gru_zero = config['target_policy'].get(
            'init_gru_weights_zero', False
        )

        # Specification for RNN inputs
        self._previous_input = config['target_policy'].get('previous_input',
                                                           True)
        self._parent_input = config['target_policy'].get('parent_input',
                                                         False)
        self._sibling_input = config['target_policy'].get('sibling_input',
                                                          False)

        # Number of training steps
        self._num_steps = config['num_steps']

        # Flag as to whether to run max likelihood or ELBO optimisation
        self._max_likelihood_flag = config.get('max_likelihood', False)

        # Information about the prior
        self._prior_mean = config.get('prior_mean', 0.0)
        self._prior_variance = config.get('prior_variance', 1.0)

        # Remove x variables as tokens
        self._remove_x_vars = config.get('remove_x_vars', False)

        # Plot if available
        self._plotting = config.get('plotting', False)

        # Track KL divergence through training
        self._track_kl_divergence = config.get('track_kl_divergence', False)

        # Flag whether to calculate posteriors at the end of training
        self._calc_posteriors_flag = config.get('calculate_posteriors', True)

        # Flag to determine which behaviour policy to use
        self._behaviour_policy_name = config.get('behaviour_policy', 'target')

        # Evidence only needs to be computed once
        self._evidence = None

        # Specifying a risk-seeking RL epsilon value turns on risk-seeking RL
        self._epsilon = config.get('risk_seeking_epsilon', None)

        # Specify lambda for max entropy reward
        # If no lambda is given, max entropy reward is not included
        self._entropy_lambda = config.get('entropy_lambda', None)

        # Specify baseline to use during reinforce
        self._baseline = config.get('baseline', {'name': 'mean'})
        if self._baseline['name'] == 'ewma':
            self._ewma = None if self._baseline.get('jumpstart', False) else 0.0
            self._ewma_alpha = self._baseline['alpha']

        # Seed random number generators
        self._seed = config.get('seed', None)
        if self._seed is not None:
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)

        self._verbosity = config.get('verbosity', 2)

    def train(self, data):

        self._initialise(data)

        if self._max_likelihood_flag:
            results = self._maximise_likelihood(data)
        else:
            results = self._maximise_elbo(data)

        if self._plotting:
            self._plot_distrs(data)

        return results

    def _maximise_likelihood(self, data):

        optimiser = torch.optim.RMSprop(self._q._net.parameters(), lr=self._lr)

        # Keep track of sampled z with the highest maximum likelihood
        r_max = None
        best_z = None

        for i in range(self._num_steps):

            # Sample z from behaviour policy
            sampled_z = \
                [self._behaviour_policy.sample_and_optimise(data, log_likelihood)
                 for i in range(self._num_eq_samples)]

            # Calculate likelihoods of sampled models
            log_likelihoods = np.array(
                [log_likelihood(data, z) for z in sampled_z]
            )

            # Set the reward to the log likelihood
            rewards = log_likelihoods

            # Keep track of best performing sample
            r_m = np.max(rewards)
            if r_max is None or r_m > r_max:
                r_max = r_m
                best_z = copy.deepcopy(sampled_z[np.argmax(rewards)])

            # Filter if using risk-seeking policy gradient
            if self._epsilon is not None:

                # Calculate quantile according to 1 - ε
                quantile = np.quantile(rewards, 1 - self._epsilon)

                # Filter equations and rewards
                filtered_z, filtered_r = zip(
                    *[(z, r) for z, r in zip(sampled_z, rewards)
                      if r >= quantile]
                )

                # Convert filtered values back to lists
                sampled_z = list(filtered_z)
                rewards = list(filtered_r)

                rewards = [r - quantile for r in rewards]

            else:

                # Calculate baseline
                baseline = None
                match self._baseline['name']:

                    # Use mean as baseline
                    case 'mean':
                        baseline = rewards.mean()

                    # Use exponentially weighted moving average as baseline
                    case 'ewma':

                        if self._ewma is None:
                            self._emwa = rewards.mean()
                        else:
                            self._emwa = (self._ewma_alpha * rewards.mean()
                                          + (1.0 - self._ewma_alpha)
                                            * self._ewma)

                        baseline = self._emwa

                # Apply baseline
                if baseline is not None:
                    rewards = rewards - baseline

            # Calculate importance weights
            importance_weights = [self._behaviour_policy.importance_weight(z)
                                  for z in sampled_z]

            # Apply importance weights
            rewards = np.array(
                [w * r for r, w in zip(rewards, importance_weights)]
            )

            '''
            for z, r, w in zip(sampled_z, rewards, importance_weights):
                print(f"{z.get_infix()}      {r}                "
                      f"{self._behaviour_policy.importance_weight(z)}")
                # for t in z.tokens():
                #     print(f"{t['op']}")
                # net_outs = self._q.net_outs(z)
                # print(net_outs)
            # exit()
            '''

            losses = torch.stack(
                [-self._q.log_pdf(z) * r for z, r in zip(sampled_z, rewards)]
            )
            loss = losses.mean()

            # Apply entropy loss if specified
            if self._epsilon is not None and self._entropy_lambda is not None:
                entropy = self.calculate_entropy(sampled_z)
                entropy_loss = -self._entropy_lambda * entropy
                loss += entropy_loss

            print('Step: {}   Loss: {}'.format(str(i), loss.item()))

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()

        '''
        sampled_z = [self._q.sample_and_optimise(data, log_likelihood)
                     for i in range(self._num_eq_samples)]
        for z in sampled_z:
            print('z: ' + z.get_infix() + '    pdf: '
                  + str(self._q.pdf(z).item()))
        '''

        all_exps = None
        if self._verbosity > 1:
            all_exps = self._enumerate_expressions(data)
            all_exps_sorted = sorted(all_exps, key=lambda z: self._q.pdf(z).item(),
                                     reverse=True)
            for i, z in enumerate(all_exps_sorted):
                print(f'{i+1}  z: {z.get_infix()}    q(z): {self._q.pdf(z).item()} '
                      f'p(x|z): {likelihood(data, z)} '
                      f'q_b(z): {self._behaviour_policy.pdf(z)}')
                '''
                print('z: ' + z.get_infix() + '    q(z): '
                      + str(self._q.pdf(z).item()) + '     p(x|z): '
                      + str(likelihood(data, z)))
                '''
                # if i > 20:
                    # break

        self._best_model = best_z
        print(f'\nBest z located: {best_z.get_infix()} simplified: '
              f'{best_z.get_infix(True)}  reward: {r_max}')

        return self._q, all_exps

    def _maximise_elbo(self, data):

        optimiser = torch.optim.RMSprop(self._q._net.parameters(), lr=self._lr)

        # Keep track of sampled z with the highest ELBO
        r_max = None
        best_z = None

        mus = []
        kl_divs = []

        for i in range(self._num_steps):

            # Sample z from behaviour policy
            sampled_z = \
                [self._behaviour_policy.sample_and_optimise(data, log_likelihood)
                 for i in range(self._num_eq_samples)]

            # Calculate ELBO
            elbos, log_likelihoods = self.elbos(data, sampled_z)

            # Track values of interest
            if self._distr_over_consts:
                mu = self._q.net_outs(sampled_z[0])[-1][-2]
                mus.append(mu)

            if self._track_kl_divergence:
                kl_divergence = self.kl_divergence(data, num_samples=100)
                kl_divs.append(kl_divergence)

            # Set reward to the elbo
            rewards = elbos

            # Keep track of best performing sample according to log likelihood
            r_m = np.max(log_likelihoods.numpy())
            if r_max is None or r_m > r_max:
                r_max = r_m
                best_z = copy.deepcopy(sampled_z[np.argmax(log_likelihoods)])

            # Subtract baseline from rewards
            baseline = rewards.mean()
            rewards = rewards - baseline

            # Calculate importance weights
            importance_weights = [self._behaviour_policy.importance_weight(z)
                                  for z in sampled_z]

            '''
            for z, r, w in zip(sampled_z, rewards, importance_weights):
                print(f"{z.get_infix()}      {r}                {w}")
            exit()
            '''

            # Apply importance weights
            rewards = np.array(
                [w * r for r, w in zip(rewards, importance_weights)]
            )

            loss = torch.stack(
                [-self._q.log_pdf(z) * r for z, r in zip(sampled_z, rewards)]
            ).mean()

            print('Step: {}   Loss: {}'.format(str(i), loss.item()))

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()

        if self._plotting:
            plt.plot(range(self._num_steps), mus, label='mu')
            plt.legend()
            plt.show()
            if self._track_kl_divergence:
                plt.plot(range(self._num_steps), kl_divs)
                plt.show()

        '''
        sampled_z = [self._behaviour_policy.sample_and_optimise(data, log_likelihood)
                     for i in range(self._num_eq_samples)]
        for z in sampled_z:
            print('z: ' + z.get_infix())
        '''

        if self._calc_posteriors_flag:

            true_posteriors, all_exps = self.posteriors(data)

            kl_divergence = self.kl_divergence(data, num_samples=1000)
            print('KL divergence:', kl_divergence)
            print('------------------------------')

        else:
            all_exps = self._enumerate_expressions(data)
            true_posteriors = [None] * len(all_exps)

        for p_z_x, z in zip(true_posteriors, all_exps):
            print(
                'z: ' + z.get_infix() + '    q(z): '
                + str(self._q.pdf(z).item()) + '    p(z|x): '
                + str(p_z_x)
            )
            consts_params = self._q.get_consts_params(z)
            print('     q consts params:', consts_params)

        # Optimise constants according to maximum likelihood and print
        # Of course, this is not necessarily the mode of the posterior but
        # if the variance of the prior is wide enough, it will be close
        all_z = self._enumerate_expressions(data)
        for z in all_z:
            z.convert_distr_to_opt_consts()
        optimised_z = [optimise_eq_consts(z, data, log_likelihood) for z in all_z]
        print('Optimised models:')
        for z in optimised_z:
            print(z.get_infix())

        # Set best model found throughout training
        self._best_model = best_z
        print(f'\nBest z located: {best_z.get_infix()} simplified: '
              f'{best_z.get_infix(True)}  reward: {r_max}')

        return self._q, true_posteriors, all_exps

    def _initialise(self, data):

        # Finish creating token set
        if not self._remove_x_vars:
            for i in range(len(data['x'][0])):
                self._token_set.append({"op": "x_" + str(i), "type": "const",
                                        "sub_type": "var_const",
                                        "id": self._token_id})
                self._token_id += 1

        # A mask to apply so that only constants are sampled
        self._consts_mask = torch.from_numpy(np.array(
            [0.0 if t['type'] == 'const' else -1e9 for t in self._token_set]
        ))

        # A mask to apply so that only unary operators and consts are sampled
        self._un_ops_consts_mask = torch.from_numpy(np.array(
            [0.0 if t['type'] == 'un_op' or t['type'] == 'const' else -1e9
             for t in self._token_set]
        ))

        # Calculate total number of models
        self._total_num_eqs = calculate_total_num_eqs(self._token_set,
                                                      self._max_num_tokens)

        # Create surrogate distribution, q, which is optimised to approximate
        # the posterior
        self._q = q(self._token_set, self._hidden_layer_size,
                    self._init_gru_zero, self._max_depth, self._max_num_tokens,
                    self._distr_over_consts, self._q_const_variance,
                    self._consts_mask, self._un_ops_consts_mask,
                    self._previous_input, self._parent_input,
                    self._sibling_input)

        # If enumerate all behaviour policy is being used, enumerate models
        # here
        all_models = None
        if self._behaviour_policy_name == 'enumerate_all':
            all_models = self._enumerate_expressions(data)

        # Use separate behaviour policy if specified
        self._behaviour_policy = BehaviourPolicy(self._behaviour_policy_name,
                                                 self._q, self._token_set,
                                                 self._max_num_tokens,
                                                 all_models)

    def _prior(self, z):

        total_num_eqs = calculate_total_num_eqs(self._token_set,
                                                self._max_num_tokens)
        # Uniform prior
        prior = 1 / total_num_eqs

        if self._distr_over_consts:
            for c in z.distr_const_tokens():
                const_prior = scipy.stats.norm.pdf(c['value'],
                                                   self._prior_mean,
                                                   self._prior_variance)
                prior *= const_prior

        return prior

    def _log_prior(self, z):
        return math.log(self._prior(z))

    # Calculate posterior for specific model z
    def posterior(self, data, z, all_z):
        return likelihood(data, z) * self._prior(z) / self.evidence(data, all_z)

    # Calculate the true posterior for all enumerated models
    def posteriors(self, data):

        # Enumerate all expressions
        all_z = self._enumerate_expressions(data)

        # Calculate p(z|x) for all expressions
        p_z_x = [self.posterior(data, z, all_z) for z in all_z]

        return p_z_x, all_z

    def evidence(self, data, zs):
        if self._evidence is None:
            self._evidence = self._calculate_evidence(data, zs)
        return self._evidence

    # Calculate p(x) (evidence) over all models, zs
    def _calculate_evidence(self, data, zs):

        num_distr_consts = [e.num_distr_consts() for e in zs]
        total_num_distr_consts = sum(num_distr_consts)

        # Calculate p(x) based on the law of total probability
        if total_num_distr_consts == 0:
            p_x = sum([likelihood(data, z) * self._prior(z) for z in zs])

        # Calculate p(x) using a numerical integrator
        else:
            # return [None] * len(all_exps), all_exps

            def joint_func(*args):

                # Unpack arguments
                num_consts = args[-1]
                cumm_num_consts = [0] + list(itertools.accumulate(num_consts))
                total_num_consts = sum(num_consts)
                x = args[:total_num_consts + 1]
                all_exps = args[total_num_consts + 1]

                # Sample a particular expression
                samp = x[0]
                idx = int(samp)

                # This might happen if the integrator samples exactly the
                # upper bound
                if idx >= len(all_exps):
                    return 0.0

                z = copy.deepcopy(all_exps[idx])

                # Parse consts relevant to selected expression
                this_z_consts = x[cumm_num_consts[idx] + 1:
                                  cumm_num_consts[idx + 1] + 1]
                other_z_consts = x[1:cumm_num_consts[idx] + 1] \
                                 + x[cumm_num_consts[idx + 1] + 1:]

                if z.num_distr_consts() > 0:
                    z.set_distr_consts(this_z_consts)

                if any(c < 0.0 or c > 1.0 for c in other_z_consts):
                    return 0.0

                return likelihood(data, z) * self._prior(z)

            # Create integration bounds
            # The first bound is for selecting the particular expression
            # The remaining bounds are for each of the optimisable constants
            integration_bounds = [[0, len(zs)]]

            for i in range(total_num_distr_consts):
                integration_bounds.append([-np.inf, np.inf])

            p_x, error = scipy.integrate.nquad(joint_func,
                                               integration_bounds,
                                               args=(zs, num_distr_consts))
        return p_x

    def log_evidence(self, data, zs):
        return math.log(self.evidence(data, zs))

    # Calculate list of values such that when you take the mean, you get the
    # ELBO.
    # Also returns sampled models
    def elbos(self, data, samples):

        # Calculate log likelihoods of sampled models
        log_likelihoods = torch.tensor(
            [log_likelihood(data, z) for z in samples],
            requires_grad=False
        )

        # Calculate log q(z) under the surrogate distribution for samples
        # models
        log_q_zs = torch.stack(
            [self._q.log_pdf(z) for z in samples]
        ).detach()

        # Calculate priors, ln p(z), for sampled models
        log_priors = torch.tensor(
            [self._log_prior(z) for z in samples],
            requires_grad=False
        )

        # Calculate and return ELBO and log likelihoods
        return log_likelihoods + log_priors - log_q_zs, log_likelihoods

    # Calculate the KL divergence between q(z) and p(z|x)
    def kl_divergence(self, data, num_samples):

        # Sample z from q
        samples = [self._behaviour_policy.sample_and_optimise(data,
                                                              log_likelihood)
                   for i in range(num_samples)]

        elbo = self.elbos(data, samples)[0].mean()

        # Enumerate all expressions
        all_z = self._enumerate_expressions(data)

        # Calculate KL divergence
        kl_divergence = self.log_evidence(data, all_z) - elbo

        return kl_divergence.item()

    # Calculate entropy of batch of sampled zs
    def calculate_entropy(self, zs):

        probs = torch.stack([self._q.pdf(z) for z in zs])
        log_probs = torch.stack([self._q.log_pdf(z) for z in zs])

        entropy = torch.sum(probs * log_probs)

        return entropy

    # Fit model for sklearn API interface
    def fit(self, X, y):
        data = {'x': X.to_numpy(), 'y': y}
        self.train(data)

    # Inference for sklearn API interface
    def predict(self, X):

        if not hasattr(self, '_best_model'):
            raise RuntimeError('Must train before performing inference')

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        y = self._best_model.evaluate(X)

        return y

    # Best model found during training
    def best_model(self):
        if not hasattr(self, '_best_model'):
            raise RuntimeError('Must train in order to get the best model')
        return self._best_model

    # Get hyperparameters of model (sklearn interface)
    def get_params(self, deep=True):
        return {}

    # Set hyperparameters of model (sklearn interface)
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # Enumerate all expressions according to a specific token set and a maximum
    def _enumerate_expressions(self, data=None):

        l_m = self._max_num_tokens

        # Split tokens by type
        consts = [t for t in self._token_set if t['type'] == 'const']
        un_ops = [t for t in self._token_set if t['type'] == 'un_op']
        bin_ops = [t for t in self._token_set if t['type'] == 'bin_op']

        # Initialize list to store expressions by length
        # expressions[0] is empty (unused), expressions[1] for length 1, etc.
        expressions = [[] for _ in range(l_m + 1)]

        # Base case: length 1 expressions are just the constants
        expressions[1] = [[copy.deepcopy(c)] for c in consts]

        # Build expressions iteratively from length 2 to l_m
        for length in range(2, l_m + 1):

            # Add expressions starting with unary operations
            # Format: [unary_op] + subexpression_of_length_(length-1)
            for uop in un_ops:
                for subexpr in expressions[length - 1]:
                    expressions[length].append(
                        [copy.deepcopy(uop)] + copy.deepcopy(subexpr)
                    )

            # Add expressions starting with binary operations (if length >= 3)
            # Format: [binary_op] + expr1 + expr2, where
            # total length = 1 + len(expr1) + len(expr2)
            if length >= 3:
                for bop in bin_ops:
                    # Split remaining tokens (length-1) between two subexpressions
                    for k in range(1, length - 1):
                        for expr1 in expressions[k]:
                            for expr2 in expressions[length - 1 - k]:
                                expressions[length].append(
                                    [copy.deepcopy(bop)] + copy.deepcopy(expr1)
                                    + copy.deepcopy(expr2)
                                )

        # Collect all expressions from length 1 to l_m
        all_expressions = [Equation(expr) for length in range(1, l_m + 1)
                           for expr in expressions[length]]

        # Check whether pre softmax masks would have been applied if these
        # expressions were sampled from q
        for e in all_expressions:
            e.apply_pre_softmax_mask(self._max_num_tokens,
                                     self._consts_mask,
                                     self._un_ops_consts_mask)

        # If we are considering a distribution over constants then set the
        # constant to the mean of the distribution
        if self._distr_over_consts:
            for exp in all_expressions:
                net_outs = self._q.net_outs(exp)
                consts = [out[-2] for out, token in zip(net_outs, exp.tokens())
                          if token['sub_type'] == 'float_const']
                exp.set_distr_consts(consts)

        # Optimise constants according to maximum likelihood if there are
        # any optimisable constants
        if data is not None:
            all_expressions = [optimise_eq_consts(eq, data, log_likelihood)
                               for eq in all_expressions]

        return all_expressions

    # Plot priors, likelihoods, joints and posterior for simplest case.
    # NOTE: This is just for testing and should not be used functionally.
    def _plot_distrs(self, data):

        all_exps = self._enumerate_expressions(data)

        if len(all_exps) > 2:
            print('Cannot plot distributions for more than y=c')
            return

        x = np.arange(-5.0, 5.0, 0.01)
        exps = [copy.deepcopy(all_exps[0]) for _ in range(len(x))]
        for val, e in zip(x, exps):
            e.set_distr_consts([val])

        priors = [self._prior(z) for z in exps]
        likelihoods = [likelihood(data, z) for z in exps]
        joints = [l * p for p, l in zip(priors, likelihoods)]
        evidence = self.evidence(data, [exps[0]])
        posteriors = [j / evidence for j in joints]
        qs = [self._q.pdf(z).item() for z in exps]

        prior_max = x[np.argmax(priors)]
        likelihood_max = x[np.argmax(likelihoods)]
        joint_max = x[np.argmax(joints)]
        posterior_max = x[np.argmax(posteriors)]
        q_max = x[np.argmax(qs)]

        print('Evidence:', evidence)
        print('Prior max:', prior_max)
        print('Likelihood max:', likelihood_max)
        print('Joint max:', joint_max)
        print('Posterior max:', posterior_max)
        print('q max:', q_max)

        plt.plot(x, priors, label='Prior')
        plt.plot(x, likelihoods, label='Likelihood')
        plt.plot(x, joints, label='Joint')
        plt.plot(x, posteriors, label='Posterior')
        plt.plot(x, qs, label='q(z)')

        plt.legend()

        plt.show()

        # Check posterior integrates to 1
        '''
        def post_func(*args):

            z = copy.deepcopy(args[1])
            z.set_distr_consts([args[0]])
            return self.posterior(data, z, [z], evidence)

        integration_bounds = [[-np.inf, np.inf]]

        out, error = scipy.integrate.nquad(post_func,
                                           integration_bounds,
                                           args=(exps[0], data, evidence))
        print(out)
        print(error)
        '''


def log_likelihood(data, z):

    means = z.evaluate(data['x'])
    log_likelihoods = [scipy.stats.norm.logpdf(y, mean)
                       for y, mean in zip(data['y'], means)]
    return sum(log_likelihoods)


def likelihood(data, z):

    means = z.evaluate(data['x'])
    likelihoods = [scipy.stats.norm.pdf(y, mean)
                   for y, mean in zip(data['y'], means)]
    return math.prod(likelihoods)


# Calculate total number of models possible according to token set and
# max number of tokens
def calculate_total_num_eqs(token_set, max_num_tokens):

    n_c = sum(1 for t in token_set if t['type'] == 'const')
    n_u = sum(1 for t in token_set if t['type'] == 'un_op')
    n_b = sum(1 for t in token_set if t['type'] == 'bin_op')
    t_max = max_num_tokens

    """
    Calculate the number of distinct expressions with <= t_max tokens.

    Parameters:
    - t_max: Maximum number of tokens (integer >= 0)
    - n_c: Number of distinct constants (integer >= 0)
    - n_u: Number of distinct unary operators (integer >= 0)
    - n_b: Number of distinct binary operators (integer >= 0)

    Returns:
    - Number of expressions with 1 to t_max tokens inclusive
    """
    if t_max < 0:
        return 0

    # b[t] stores number of expressions with exactly t tokens
    b = [0] * (t_max + 1)
    if t_max >= 1:
        b[1] = n_c

    # Compute exact counts for each expression size
    for t in range(2, t_max + 1):
        unary = n_u * b[t - 1]
        binary = 0
        for i in range(1, t - 1):
            binary += b[i] * b[t - 1 - i]
        b[t] = unary + n_b * binary

    # Sum up to t_max
    return sum(b[:t_max + 1])
