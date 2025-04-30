# Surrogate distribution, q, which is optimised to approximate the
# posterior.
# It currently consists of a recurrent neural network that outputs
# a sequence of categorical distribution parameters.

import torch
import copy
import numpy as np
from .vicatsr_net import NN
from .equation import Equation, optimise_eq_consts
from util.tree import get_parent, get_sibling
from .net_masks import NetMasks


class q:

    def __init__(self, token_set, max_depth, max_num_tokens,
                 distr_over_consts, const_variance,
                 net_masks, previous_input: bool,
                 parent_input: bool, sibling_input: bool,
                 hidden_layer_size: int = None,
                 init_gru_zero: bool = False,
                 net_path: str = None):

        self._max_depth = max_depth
        self._max_num_tokens = max_num_tokens
        self._token_set = token_set
        self._distr_over_consts = distr_over_consts

        self._net_masks = net_masks

        # Variance for normal distribution over constants
        # If set to None, this is also optimised
        self._const_variance = const_variance

        # Determines what is input into the network
        self._previous_input = previous_input
        self._parent_input = parent_input
        self._sibling_input = sibling_input

        rnn_input_size = sum([self._previous_input, self._parent_input,
                              self._sibling_input]) * len(token_set)

        # Read net from file
        if net_path:
            self._net = NN.load(net_path)

        # Create recurrent neural network
        else:
            self._net = NN(rnn_input_size, len(token_set),
                           hidden_layer_size, distr_over_consts,
                           True if const_variance is None else False,
                           init_gru_zero)

    def sample(self):

        self._net.reset(1)

        # Loop until max depth or sufficient number of constants have been
        # sampled
        tokens = []
        num_consts_required = 1

        while num_consts_required > 0:

            pre_softmax_mask = self.determine_pre_softmax_mask(
                tokens, num_consts_required
            )

            net_input = self.get_net_inputs(tokens)

            # Pass input through network
            out = self._net.forward(net_input, pre_softmax_mask).detach().numpy()

            # Sample token from categorical distribution
            token = copy.deepcopy(
                np.random.choice(
                    self._token_set, 1, p=out[:len(self._token_set)]
                )[0]
            )

            # If token is distr_const and distribution over constants is on
            # then sample value from distribution
            if self._distr_over_consts and token['op'] == 'distr_const':

                # Variance of const distribution is either given in config
                # or optimised
                const_variance = out[-1] if self._const_variance is None \
                                         else self._const_variance
                # Sample
                token['value'] = np.random.normal(loc=out[-2],
                                                  scale=const_variance)

            # Increase or decrease the number of constants required
            # depending on the sample token type
            if token['type'] == 'bin_op':
                num_consts_required += 1
            elif token['type'] == 'const':
                num_consts_required -= 1

            token['pre_softmax_mask'] = copy.deepcopy(pre_softmax_mask)

            tokens.append(token)

        return Equation(tokens)

    # Sample from q and also optimise const tokens of sampled equation
    def sample_and_optimise(self, data, log_likelihood_func):
        return optimise_eq_consts(self.sample(), data, log_likelihood_func)

    # Calculate probability of an equation, z, under q
    def pdf(self, z):

        self._net.reset(1)

        probs = []
        for i, t in enumerate(z.tokens()):

            net_inputs = self.get_net_inputs(z.tokens()[:i])
            out = self._net.forward(net_inputs, t['pre_softmax_mask'])

            # Generate one hot vector for current token
            one_hot = torch.zeros(len(self._token_set))
            one_hot[t['id']] = 1.0

            probs.append(torch.sum(out[:len(self._token_set)] * one_hot))

            if self._distr_over_consts and t['sub_type'] == 'float_const':

                # Variance of const distribution is either given in config
                # or optimised
                const_variance = out[-1] if self._const_variance is None \
                                         else self._const_variance

                probs.append(torch.exp(torch.distributions.normal.Normal(
                    loc=out[-2], scale=const_variance
                ).log_prob(torch.tensor(t['value']))))

        return torch.prod(torch.stack(probs))

    # Calculate log probability of an equation, z, under q
    def log_pdf(self, z):

        self._net.reset(1)

        log_probs = []
        for i, t in enumerate(z.tokens()):

            net_inputs = self.get_net_inputs(z.tokens()[:i])
            out = self._net.forward(net_inputs, t['pre_softmax_mask'])

            # Generate one hot vector for current token
            one_hot = torch.zeros(len(self._token_set))
            one_hot[t['id']] = 1.0

            log_probs.append(torch.log(
                torch.sum(out[:len(self._token_set)] * one_hot)
            ))

            if self._distr_over_consts and t['sub_type'] == 'float_const':

                # Variance of const distribution is either given in config
                # or optimised
                const_variance = out[-1] if self._const_variance is None \
                                         else self._const_variance

                log_probs.append(torch.distributions.normal.Normal(
                    loc=out[-2], scale=const_variance
                ).log_prob(torch.tensor(t['value'])))

        return torch.sum(torch.stack(log_probs))

    # Get all network outputs for a particular equation
    def net_outs(self, z):

        self._net.reset(1)

        net_outs = []
        for i, t in enumerate(z.tokens()):

            net_inputs = self.get_net_inputs(z.tokens()[:i])

            out = self._net.forward(net_inputs, t['pre_softmax_mask'])
            net_outs.append(out)

        return torch.stack(net_outs).detach().numpy()

    # Determine pre softmax mask to apply
    def determine_pre_softmax_mask(self, tokens, num_consts_required):

        masks = []

        # Apply mask to only sample unary operators and constants
        if self._max_num_tokens - len(tokens) == num_consts_required + 1:
            masks = ['un_ops', 'consts']

        # Apply mask to only sample constants
        if self._max_num_tokens - len(tokens) < num_consts_required + 1:
            masks = ['consts']

        # Check whether x - x will occur, this is obviously superflous and
        # helps to prevent divide by 0 errors
        remove_var = None
        '''
        parent = get_parent(tokens)
        if parent and parent['op'] == '-':
            sibling = get_sibling(tokens)
            if sibling and sibling['sub_type'] == 'var_const':
                remove_var = [sibling['op']]
        '''

        return self._net_masks.compose_mask(masks, remove_var)

    # Get means and variances output by the network for all constants in
    # an equation
    def get_consts_params(self, z):
        return [[out[-2], out[-1]]
                for out, token in zip(self.net_outs(z), z.tokens())
                if token['op'] == 'distr_const']

    @property
    def net_masks(self):
        return self._net_masks

    # Calculate network inputs
    def get_net_inputs(self, tokens):

        if len(tokens) == 0:
            return torch.zeros(self._net.num_inputs())
        else:

            inputs = []

            if self._previous_input:
                x = torch.zeros(len(self._token_set))
                x[tokens[-1]['id']] = 1.0
                inputs.append(x)

            if self._parent_input:
                parent = get_parent(tokens)
                x = torch.zeros(len(self._token_set))
                if parent:
                    x[parent['id']] = 1.0
                inputs.append(x)

            if self._sibling_input:
                sibling = get_sibling(tokens)
                x = torch.zeros(len(self._token_set))
                if sibling:
                    x[sibling['id']] = 1.0
                inputs.append(x)

            return torch.cat(inputs)

        # TODO: Might have to input sampled value for constants back into
        # the network

    def to_json(self):

        j = self.__dict__

        # Remove '_' prefix from all keys
        j = {k.lstrip('_'): v for k, v in j.items()}

        # net_masks not needed
        del j['net_masks']

        return j

    @classmethod
    def from_json(cls, j):

        # Create NetMasks object from saved token set
        j['net_masks'] = NetMasks(j['token_set'])

        return cls(**j)
