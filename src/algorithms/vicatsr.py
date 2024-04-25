# Variational Inference Symbolic Regression using categorical variables
# over tokens.

# This algorithm is similar to Deep Symbolic Regression in that it uses a
# sequence of categorical variables over a set of tokens to build an
# analytic equation.

from algorithms.algorithm import Algorithm
import numpy as np
import scipy
import copy
import torch
torch.set_default_dtype(torch.float64)


class VICatSR(Algorithm):

    def __init__(self, config):

        # Prepare binary and unary operations as tokens
        self._token_set = []

        for bo in config['operators']['binary_ops']:
            self._token_set.append({"op": bo, "type": "bin_op"})
        for uo in config['operators']['unary_ops']:
            self._token_set.append({"op": uo, "type": "un_op"})

        # Add constant as token
        self._token_set.append({"op": 1.0, "type": "const"})

        # Number of equations sampled to calculate expected loss
        self._num_eq_samples = config['num_eq_samples']

        # Maximum equation tree depth
        self._max_depth = config['max_depth']

        # Learning rate for ADAM optimiser
        self._lr = config['learning_rate']

        # Number of training steps
        self._num_steps = config['num_steps']

        # Seed random number generators
        self._seed = config.get('seed', None)
        if self._seed is not None:
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)

    def train(self, data):

        # Initialise neural network and token set according to data
        self._initialise(data)

        for i in range(self._num_steps):

            self._optimiser.zero_grad()

            # Calculate loss
            loss = self._calculate_loss(data)

            # Optimise
            loss.backward()
            self._optimiser.step()

            print('Loss: ', loss.item())

    def _initialise(self, data):

        # Finish creating token set
        for i in range(len(data['x'][0])):
            self._token_set.append({"op": "x_" + str(i), "type": "const"})

        # Create surrogate distribution, q, which is optimised to approximate
        # the posterior
        self._q = q(self._token_set, self._max_depth)

        # Create ADAM optimiser
        self._optimiser = torch.optim.Adam(self._q._net.parameters(),
                                           lr=self._lr)

    def _calculate_loss(self, data):

        # Sample equations from q(z)
        sampled_eqs = [self._q.sample() for i in range(self._num_eq_samples)]

        # Calculate ELBO
        elbo = self._calculate_elbo(data, sampled_eqs)

        return -torch.mean(elbo)

    def _calculate_elbo(self, data, z):

        # Prior
        prior = torch.from_numpy(self._evaluate_prior(z))

        # Likelihood
        likelihood = torch.from_numpy(self._evaluate_likelihood(data, z))

        # Calculate q(z)
        q_z = self._q.pdf(z, self._token_set)

        # Calculate ELBO
        elbo = torch.log(prior) + torch.log(likelihood) - torch.log(q_z)

        return elbo

    def _evaluate_prior(self, z):

        # For now, the prior is just the uniform distribution
        return np.array([1 / len(self._token_set) ** e.num_tokens() for e in z])

    def _evaluate_likelihood(self, data, z):

        likelihoods = []
        for eq in z:
            likelihood = 1.0
            for i in range(len(z)):
                likelihood *= scipy.stats.norm.pdf(data['y'][i],
                                                   loc=z[i].evaluate(data['x']),
                                                   scale=1.0)
            likelihoods.append(likelihood)

        return np.array(likelihoods)


# Surrogate distribution, q, which is optimised to approximate the
# posterior.
# It currently consists of a recurrent neural network that outputs
# a sequence of categorical distribution parameters.
class q:

    def __init__(self, token_set, max_sampling_depth):

        # Create recurrent neural network
        self._net = NN(len(token_set), len(token_set), 256)

        self._max_depth = max_sampling_depth
        self._token_set = token_set

    def sample(self):

        self._net.reset(1)

        # Loop until max depth or sufficient number of constants have been
        # sampled
        tokens = []
        x = torch.zeros(self._net.num_inputs())
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

            tokens.append(token)
            i += 1

        return Equation(tokens)

    # Calculate probabilities of a set of equations, z, under q
    def pdf(self, z, token_set):

        probabilities = []

        for eq in z:

            self._net.reset(1)

            x = torch.zeros(self._net.num_inputs())

            prob = 1.0
            for t in eq.tokens():

                x = self._net.forward(x)

                # Generate one hot vector for current token
                idx = token_set.index(t)
                one_hot = torch.zeros(self._net.num_inputs())
                one_hot[idx] = 1.0

                prob *= torch.sum(x * one_hot)

            probabilities.append(prob)

        return torch.stack(probabilities)


class NN(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size):
        super().__init__()

        self._hidden_size = hidden_size

        self._l1 = torch.nn.GRUCell(num_inputs, self._hidden_size)
        self._l2 = torch.nn.Linear(self._hidden_size, num_outputs)

        self._num_inputs = num_inputs

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

    def num_inputs(self):
        return self._num_inputs


# A class for representing analytic equations and evaluating them
class Equation:

    # Tokens should be input in polish notation
    def __init__(self, tokens):

        # Equation is represented in polish notation
        self._eq = tokens

    # Evaluate equation according to data variable values, x.
    def evaluate(self, x):

        # Substitute variables for data, x
        eq = copy.deepcopy(self._eq)
        for token in eq:
            for i in range(x.shape[0]):
                if token['op'] == ('x_' + str(i)):
                    token['op'] = x[:, i]

        # Convert Polish notation to Reverse Polish Notation
        eq.reverse()

        # Evaluate equation using stack
        stack = []

        for t in eq:

            # If token is a constant, push onto stack
            if t['type'] == 'const':
                stack.append(t['op'])

            # Otherwise apply operators to elements on the stack
            else:

                match t['op']:
                    case '*':
                        stack.append(stack.pop() * stack.pop())
                    case '/':
                        stack.append(stack.pop() / stack.pop())
                    case '+':
                        stack.append(stack.pop() + stack.pop())
                    case '-':
                        stack.append(stack.pop() - stack.pop())
                    case 'cos':
                        stack.append(np.cos(stack.pop()))
                    case 'sin':
                        stack.append(np.sin(stack.pop()))
                    case 'exp':
                        stack.append(np.exp(stack.pop()))
                    case _:
                        raise RuntimeError(t['op']
                                           + ' is not a recognised operator')

        return stack.pop()

    def num_tokens(self):
        return len(self._eq)

    def tokens(self):
        return self._eq
