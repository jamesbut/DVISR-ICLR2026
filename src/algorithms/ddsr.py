# Differentiable Deep Symbolic Regression

# This is a modification of Deep Symbolic Regression that is end-to-end
# differentiable and therefore trained in a supervised manner, not by using
# reinforcement learning.

from algorithms.algorithm import Algorithm

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

        # TODO: Maybe some kind of normalisation here

        # Gumbel-softmax layer
        x = torch.nn.functional.gumbel_softmax(x)

        return x

    def reset(self, batch_size):
        self._hx = torch.zeros(self._hidden_size)


class DDSR(Algorithm):

    def __init__(self, config):

        # Prepare binary and unary operations as tokens
        self._token_set = []

        for bo in config['operators']['binary_ops']:
            self._token_set.append({"op": bo, "type": "bin_op"})
        for uo in config['operators']['unary_ops']:
            self._token_set.append({"op": uo, "type": "un_op"})

        # Add constant as token
        self._token_set.append({"op": "1", "type": "const"})

    def train(self, data):

        # Initialise neural network and token set according to data
        self._initialise(data)

        x = torch.as_tensor(data['x'])

        self._net.reset(1)

        # Create symbolic expression by sampling tokens from network
        eq = DDSR._sample_eq(self._net, self._token_set)

        # Calculate distance between data and expression

        # Backpropogate error

        # Apply optimiser

    def _initialise(self, data):

        # Finish creating token set
        for i in range(len(data['x'][0])):
            self._token_set.append({"op": "x_" + str(i), "type": "const"})

        # Create recurrent neural network
        self._net = NN(len(self._token_set), len(self._token_set), 256)

    # Samples equation from network
    def _sample_eq(net, token_set):

        # Loop until max depth or terminal token is sampled
        tokens = []
        x = torch.zeros(len(token_set))
        for i in range(3):
            x = net.forward(x)
            token_idx = arg_max(x)
            print(x)


    def _build_token_set(operators):
        pass
