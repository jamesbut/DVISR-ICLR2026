# Behaviour policy in order to do off-policy RL
# i.e. when the behaviour policy is different to the target policy.

from .equation import Equation, optimise_eq_consts
import copy
import numpy as np
import math
import scipy
import random
from enum import Enum, auto


class BPStrategy(Enum):
    target = auto()
    enumerate_all = auto()
    equal_prob_tokens = auto()


class BehaviourPolicy:

    def __init__(self, name, target_policy, token_set, max_num_tokens,
                 all_eqs=None):

        if name not in BPStrategy.__members__.keys():
            raise KeyError(f'{name} not a behaviour policy strategy')

        self._strategy = BPStrategy[name]

        # All enumerated equations need to be provided for the 'enumerate_all'
        # behaviour policy
        if name == 'enumerate_all' and all_eqs is None:
            raise ValueError('Must provided enumerate equations in order to '
                             'use the enumerate_all behaviour strategy')
        else:
            self._all_eqs = all_eqs

        # Enumerate all policy cannot be used if there are distr_consts in the
        # token set
        if sum(1 for t in token_set if t['op'] == 'dist_const') > 0:
            raise ValueError('Enumerate all policy cannot be used if there '
                             'are dist_const in the token set')

        self._target_policy = target_policy

        self._token_set = token_set
        self._max_num_tokens = max_num_tokens

        # Precompute and store number of constants and unary operators
        self._num_consts = sum(1 if t['type'] == 'const' else 0
                               for t in token_set)
        self._num_unary_ops = sum(1 if t['type'] == 'un_op' else 0
                                  for t in token_set)

    def sample(self):

        # Sample according to the particular behavioural policy
        if self._strategy == BPStrategy.target:
            return self._target_policy.sample()
        elif self._strategy == BPStrategy.enumerate_all:
            return self._sample_enum_all()
        elif self._strategy == BPStrategy.equal_prob_tokens:
            return self._sample_eq_prob_tokens()

    def sample_and_optimise(self, data, log_likelihood_func):
        return optimise_eq_consts(self.sample(), data, log_likelihood_func)

    # Determine the pdf of a particular equation under the behavioural policy
    def pdf(self, z):

        if self._strategy == BPStrategy.target:
            return self._target_policy.pdf(z)
        elif self._strategy == BPStrategy.enumerate_all:
            return self._pdf_enum_all(z)
        elif self._strategy == BPStrategy.equal_prob_tokens:
            return self._pdf_eq_prob_tokens(z)

    # Determine importance weight for a particular equation
    def importance_weight(self, z):

        # If target policy is used as the behavioural policy importance weight
        # is 1.0
        if self._strategy == BPStrategy.target:
            return 1.0

        return (self._target_policy.pdf(z) / self.pdf(z)).item()

    # Sample from behavioural policy that assigns an equal probability to all
    # tokens for each step of the equation building
    def _sample_eq_prob_tokens(self):

        tokens = []
        num_consts_required = 1

        while num_consts_required > 0:

            # Determine parameters of categorical distribution
            p, pre_softmax_mask = self._determine_p(num_consts_required,
                                                    len(tokens))

            # Sample token
            token = copy.deepcopy(np.random.choice(self._token_set, 1, p=p)[0])

            token['pre_softmax_mask'] = copy.deepcopy(pre_softmax_mask)

            tokens.append(token)

            # Increase or decrease the number of constants required
            # depending on the sample token type
            if token['type'] == 'bin_op':
                num_consts_required += 1
            elif token['type'] == 'const':
                num_consts_required -= 1

        eq = Equation(tokens)

        # Sample constant values for constants with distributions
        consts_params = self._target_policy.get_consts_params(eq)
        consts = [np.random.normal(p[0], p[1]) for p in consts_params]

        eq.set_distr_consts(consts)

        return eq

    # Samples uniformly according to all enumerated equations
    def _sample_enum_all(self):

        eq = copy.deepcopy(random.choice(self._all_eqs))

        # Sample distributional constants from target policy
        consts_params = self._target_policy.get_consts_params(eq)
        consts = [np.random.normal(p[0], p[1]) for p in consts_params]

        eq.set_distr_consts(consts)

        return eq

    # Determine pdf according to the behaviour policy where an equal
    # probability is assigned to each token at each step of the equation
    # building
    def _pdf_eq_prob_tokens(self, z):

        pdfs = []
        num_consts_required = 1

        distr_const_idx = 0
        consts_params = self._target_policy.get_consts_params(z)

        for i, t in enumerate(z.tokens()):

            # Determine parameters of categorical distribution
            p, _ = self._determine_p(num_consts_required, i)

            # Determine probability of selecting token
            pdfs.append(p[t['id']])

            # If const was sampled from distribution
            if t['op'] == 'distr_const':
                pdfs.append(
                    scipy.stats.norm.pdf(
                        t['value'],
                        consts_params[distr_const_idx][0],
                        consts_params[distr_const_idx][1])
                )
                distr_const_idx += 1

            # Increase or decrease the number of constants required
            # depending on the sample token type
            if t['type'] == 'bin_op':
                num_consts_required += 1
            elif t['type'] == 'const':
                num_consts_required -= 1

        return math.prod(pdfs)

    # Determines pdf of policy that samples uniformly over all models
    def _pdf_enum_all(self, z):
        return 1.0 / len(self._all_eqs)

    # Also returns pre_softmax_mask for token
    def _determine_p(self, num_consts_required, num_sampled_tokens):

        # The default case - sample from all tokens
        p = [1 / len(self._token_set)] * len(self._token_set)
        pre_softmax_mask = None

        # Only sample unary operators and constants
        if self._max_num_tokens - num_sampled_tokens <= num_consts_required + 1:
            prob = 1 / (self._num_consts + self._num_unary_ops)
            p = [prob if t['type'] == 'const' or t['type'] == 'un_op' else 0.0
                 for t in self._token_set]
            pre_softmax_mask = self._target_policy.net_masks.un_ops_consts_mask

        # Only sample constants
        if self._max_num_tokens - num_sampled_tokens <= num_consts_required:
            prob = 1 / self._num_consts
            p = [prob if t['type'] == 'const' else 0.0
                 for t in self._token_set]
            pre_softmax_mask = self._target_policy.net_masks.consts_mask

        return p, pre_softmax_mask
