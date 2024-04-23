# Variational Inference Symbolic Regression using categorical variables
# over tokens.

# This algorithm is similar to Deep Symbolic Regression in that it uses a
# sequence of categorical variables over a set of tokens to build an
# analytic equation.

from algorithms.algorithm import Algorithm
import numpy as np


class VICatSR(Algorithm):

    def __init__(self, config):

        # Prepare binary and unary operations as tokens
        self._token_set = []

        for bo in config['operators']['binary_ops']:
            self._token_set.append({"op": bo, "type": "bin_op"})
        for uo in config['operators']['unary_ops']:
            self._token_set.append({"op": uo, "type": "un_op"})

        # Add constant as token
        self._token_set.append({"op": "1", "type": "const"})

        # Number of equations sampled to calculate expected loss
        self._num_eq_samples = config['num_eq_samples']

        # Maximum equation tree depth
        self._max_depth = config['max_depth']

    def train(self, data):

        # Initialise neural network and token set according to data
        self._initialise(data)

        x = torch.as_tensor(data['x'])

        self._net.reset(1)

        print(self._token_set)

        # Sample equations from q(z)
        sampled_eqs = self._sample_eqs()

    def _initialise(self, data):

        # Finish creating token set
        for i in range(len(data['x'][0])):
            self._token_set.append({"op": "x_" + str(i), "type": "const"})

        # Create recurrent neural network
        self._net = NN(len(self._token_set), len(self._token_set), 256)

    def _sample_eqs(self):

        eqs = []

        for i in range(self._num_eq_samples):
            eqs.append(self._sample_eq())

        return eqs

    def _sample_eq(self):

        # Loop until max depth or sufficient number of constants have been
        # sampled
        tokens = []
        x = torch.zeros(len(self._token_set))
        i = 0
        num_consts_required = 1

        while i < self._max_depth and num_consts_required > 0:

            x = self._net.forward(x)

            token = np.random.choice(self._token_set, 1,
                                     p=x.detach().numpy())[0]

            # Increase or decrease the number of constants required
            # depending on the sample token type
            match token['type']:
                case 'bin_op':
                    num_consts_required += 1
                case 'const':
                    num_consts_required -= 1

            '''
            print(x)
            print(token)
            print(num_consts_required)
            '''

            #token_idx = torch.argmax(x)

            #print(token_idx)
            #print(self._token_set[token_idx])

            tokens.append(token)
            i += 1

        return Equation(tokens)


import torch
torch.set_default_dtype(torch.float64)


class NN(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size):
        super().__init__()

        self._hidden_size = hidden_size

        self._l1 = torch.nn.GRUCell(num_inputs, self._hidden_size)
        self._l2 = torch.nn.Linear(self._hidden_size, num_outputs)

    def forward(self, x):

        # Check hidden state has been initialised
        if not hasattr(self, '_hx'):
            raise RuntimeError('Must call reset() before forward()')

        # GRU layer
        x = self._l1(x, self._hx)
        self._hx = x

        # Linear layer
        x = self._l2(x)

        # Softmax layer
        x = torch.nn.functional.softmax(x)

        # Gumbel-softmax layer
        # x = torch.nn.functional.gumbel_softmax(x)

        return x

    def reset(self, batch_size):
        self._hx = torch.zeros(self._hidden_size)


# A class for representing analytic equations and evaluating them
class Equation:

    # Tokens should be input in polish notation
    def __init__(self, tokens):

        # Equation is represented in polish notation
        self._eq = tokens

    # Evaluate equation according to variable values, x.
    # e.g. x = {'x_1': 2.3, 'x_2': -4.5}
    def evaluate(self, x: dict):

        # Substitute variables
        eq = []
        for t in self._eq:
            for v in x:
                if t == v:
                    t = x['v']
            eq.append(t)

        # TODO: Evaluate equation
