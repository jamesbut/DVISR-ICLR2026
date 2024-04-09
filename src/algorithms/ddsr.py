# Differentiable Deep Symbolic Regression

# This is a modification of Deep Symbolic Regression that is end-to-end
# differentiable and therefore trained in a supervised manner, not by using
# reinforcement learning.

from algorithms.algorithm import Algorithm
from utils.neural_network import NeuralNetwork


class DDSR(Algorithm):

    def __init__(self, config):

        # Define token set
        self._token_set_size = 5

        # Create recurrent neural network
        # TODO: make recurrent
        self._net = NeuralNetwork(
            num_inputs=1,
            num_outputs=self._token_set_size,
            num_hidden_layers=config['network']['num_hidden_layers'],
            neurons_per_hidden_layer=config['network']['neurons_per_hidden_layer'])

    def train(self, data):

        outs = self._net.forward(data['x'])
        print(outs)
