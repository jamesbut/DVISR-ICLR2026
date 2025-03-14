# Behaviour policy in order to do off-policy RL
# i.e. when the behaviour policy is different to the target policy.

from .equation import Equation, optimise_eq_consts
import copy
import numpy as np
import math
import scipy


class BehaviourPolicy:

    def __init__(self, target_policy, token_set, max_num_tokens,
                 use_target_policy=False):

        # Use target policy as behavioural policy
        self._use_target_policy = use_target_policy
        self._target_policy = target_policy

        self._token_set = token_set
        self._max_num_tokens = max_num_tokens

        # Precompute and store number of constants and unary operators
        self._num_consts = sum(1 if t['type'] == 'const' else 0
                               for t in token_set)
        self._num_unary_ops = sum(1 if t['type'] == 'un_op' else 0
                                  for t in token_set)

    def sample(self):

        if self._use_target_policy:
            return self._target_policy.sample()
        else:
            return self._sample()

    def sample_and_optimise(self, data, log_likelihood_func):
        return optimise_eq_consts(self.sample(), data, log_likelihood_func)

    # Determine the pdf of a particular equation under the behavioural policy
    def pdf(self, z):
        if self._use_target_policy:
            return self._target_policy.pdf(z)
        else:
            return self._pdf(z)

    # Determine importance weight for a particular equation
    def importance_weight(self, z):

        # TODO: Remove
        # When I uncomment this the posterior and q line up perfectly for
        # the last test, otherwise the the q is much bigger, why?
        # return 1.0

        # If target policy is used as the behavioural policy importance weight
        # is 1.0
        if self._use_target_policy:
            return 1.0

        return (self._target_policy.pdf(z) / self._pdf(z)).item()

    # Sample from this behavioural policy
    def _sample(self):

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
            match token['type']:
                case 'bin_op':
                    num_consts_required += 1
                case 'const':
                    num_consts_required -= 1

        eq = Equation(tokens)

        # Sample constant values for constants with distributions
        consts_params = self._target_policy.get_consts_params(eq)
        consts = [np.random.normal(p[0], p[1]) for p in consts_params]

        eq.set_distr_consts(consts)

        return eq

    # Determine pdf of using this behavioural policy
    def _pdf(self, z):

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
            match t['type']:
                case 'bin_op':
                    num_consts_required += 1
                case 'const':
                    num_consts_required -= 1

        return math.prod(pdfs)

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
            pre_softmax_mask = self._target_policy.un_ops_consts_mask()

        # Only sample constants
        if self._max_num_tokens - num_sampled_tokens <= num_consts_required:
            prob = 1 / self._num_consts
            p = [prob if t['type'] == 'const' else 0.0
                 for t in self._token_set]
            pre_softmax_mask = self._target_policy.consts_mask()

        return p, pre_softmax_mask
