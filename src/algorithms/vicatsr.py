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
        self._token_id = 0

        if 'binary_ops' in config['operators']:
            for bo in config['operators']['binary_ops']:
                self._token_set.append({"op": bo, "type": "bin_op",
                                        "id": self._token_id})
                self._token_id += 1

        if 'unary_ops' in config['operators']:
            for uo in config['operators']['unary_ops']:
                self._token_set.append({"op": uo, "type": "un_op",
                                        "id": self._token_id})
                self._token_id += 1

        # Add constant as token
        self._token_set.append({"op": 1.0, "type": "const",
                                "id": self._token_id})
        self._token_id += 1

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


        # TODO: Remove

        x = torch.Tensor([1.2])

        p = torch.nn.Parameter(torch.Tensor([0.0, 0.0]))
        #print('p:', p)

        #opt = torch.optim.Adam([p], lr=1e-2)
        opt = torch.optim.SGD([p], lr=1e-3)

        for i in range(100):

            z = []
            for i in range(100):
                z.append(torch.nn.functional.gumbel_softmax(p, hard=True))
            z = torch.stack(z)

            # 'reward' for each z instance
            r = torch.zeros_like(z)
            r[:, 1] = 1.0

            means = torch.sum(r * z, dim=1)

            #log_likelihood = torch.nn.Parameter(torch.Tensor([0.0]))
            log_likelihood = 0.0
            for i in range(len(means)):
                #l = torch.Tensor([-0.9389]) if means[i] == 1.0 else torch.Tensor([-1.6389])
                l = torch.distributions.normal.Normal(means[i], 1.0).log_prob(x)
                #print('l:', l)
                #log_likelihood += l
                log_likelihood = log_likelihood + l

            #q_z = torch.sum(z * torch.nn.functional.softmax(p), axis=1)
            #log_q_z = torch.sum(torch.log(q_z))
            q_z = torch.sum(z * torch.nn.functional.softmax(p))
            print(q_z)

            #print(log_q_z)

            '''
            print('LL:', log_likelihood)
            print('Lqz:', torch.log(q_z))
            print('Lqz:', log_q_z)
            #exit(0)
            '''
            #log_likelihood = log_likelihood * 10

            prior = torch.sum(z * torch.Tensor([0.5, 0.5]), axis=1)
            log_prior = torch.sum(torch.log(prior))
            #print(prior)
            #print(log_prior)

            #print(log_likelihood)
            #print(log_prior)
            #exit(0)

            #loss = -(log_likelihood - torch.log(q_z))
            #loss = -log_likelihood
            #loss = -(log_likelihood - log_q_z)
            #loss = -(log_likelihood + log_prior - log_q_z)
            #loss = -(log_likelihood + log_prior - torch.log(q_z))
            #loss = -(log_likelihood + log_prior)
            #loss = -(log_prior)
            #loss = -(log_q_z)
            loss = -(q_z)

            # WHAT IS GOING ON?

            print('Loss:', loss.item())
            print('Probs:', torch.nn.functional.softmax(p))

            loss.backward()
            opt.step()

            #print(p.grad)
            #exit(0)
            #print(p)

        exit(0)

        # Initialise neural network and token set according to data
        self._initialise(data)

        for i in range(self._num_steps):

            self._optimiser.zero_grad()

            # Calculate loss
            loss = self._calculate_loss(data)

            #print('Loss:', loss)

            # Optimise
            loss.backward()

            '''
            print('GRADS:')
            for p in self._q._net.parameters():
                print(p.grad)
            print('----------')
            print('PARAMS:')
            for p in self._q._net.parameters():
                print(p)
            print('----------')
            '''

            self._optimiser.step()


            '''
            print('PARAMS:')
            for p in self._q._net.parameters():
                print(p)
            print('----------')
            '''
            #exit(0)

            print('Step: ' + str(i) + '    Loss: ' + str(loss.item()))

    def _initialise(self, data):

        # Finish creating token set
        for i in range(len(data['x'][0])):
            self._token_set.append({"op": "x_" + str(i), "type": "const",
                                    "id": self._token_id})
            self._token_id += 1

        # Create surrogate distribution, q, which is optimised to approximate
        # the posterior
        self._q = q(self._token_set, self._max_depth)

        # Create ADAM optimiser
        #self._optimiser = torch.optim.Adam(self._q._net.parameters(),
        #                                   lr=self._lr)
        self._optimiser = torch.optim.SGD(self._q._net.parameters(),
                                           lr=self._lr)

    def _calculate_loss(self, data):

        # Sample equations from q(z)
        sampled_eqs = [self._q.sample() for i in range(self._num_eq_samples)]

        #print(sampled_eqs[0])
        #print(sampled_eqs[0].get_infix())
        #for e in sampled_eqs:
        #    print(e.get_infix())

        # Calculate ELBO
        elbo = self._calculate_elbo(data, sampled_eqs)

        #print('ELBO:', elbo)
        #exit(0)

        return -torch.mean(elbo)

    def _calculate_elbo(self, data, z):

        # Prior
        prior = self._evaluate_prior(z)
        log_prior = torch.from_numpy(np.log(prior))

        # Likelihood
        likelihood = self._evaluate_likelihood(data, z)
        log_likelihood = torch.from_numpy(np.sum(np.log(likelihood), axis=1))
        #print('Log likelihood mean:', torch.mean(log_likelihood))
        #print('Log likelihood:', log_likelihood)

        # Calculate q(z)
        q_z = self._q.pdf(z, self._token_set)
        log_q_z = torch.log(q_z)
        #print('Log q_(z):', log_q_z)
        #print(torch.mean(log_q_z))
        #exit(0)

        '''
        print('Prior:', prior)
        print('Log prior:', log_prior)
        print('Likelihood:', likelihood)
        print('Log likelihood:', log_likelihood)
        print('q(z):', q_z)
        print('Log q(z):', log_q_z)

        print('Likelihood:', likelihood)
        print('Log likelihood:', log_likelihood)
        '''

        # Calculate ELBO
        #elbo = log_prior + log_likelihood - log_q_z
        elbo = log_likelihood - log_q_z

        return elbo

    def _evaluate_prior(self, z):

        # For now, the prior is just the uniform distribution
        return np.array([1 / len(self._token_set) ** e.num_tokens() for e in z])

    def _evaluate_likelihood(self, data, z):

        likelihoods = []
        for eq in z:
            likelihood = []
            means = eq.evaluate(data['x'])
            for i in range(len(means)):
                likelihood.append(scipy.stats.norm.pdf(data['y'][i],
                                                       loc=means[i],
                                                       scale=1.0))
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

        # A mask to apply so that only constants are sampled
        self._consts_mask = []
        for t in self._token_set:
            if t['type'] == 'const':
                self._consts_mask.append(0.0)
            else:
                self._consts_mask.append(-1e9)
        self._consts_mask = torch.from_numpy(np.array(self._consts_mask))

    def sample(self):

        self._net.reset(1)

        # Loop until max depth or sufficient number of constants have been
        # sampled
        tokens = []
        x = torch.zeros(self._net.num_inputs())
        i = 0
        num_consts_required = 1

        while num_consts_required > 0:

            # Apply mask to only sample constants if more constants are needed
            # to produce a valid equation

            if self._max_depth - len(tokens) <= num_consts_required + 1:
                pre_softmax_mask = self._consts_mask
            else:
                pre_softmax_mask = None

            x = self._net.forward(x, pre_softmax_mask)
            #print('SAMPLE:', x)
            #x = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
            #x = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])

            token = copy.deepcopy(np.random.choice(self._token_set, 1,
                                                   p=x.detach().numpy())[0])

            # Increase or decrease the number of constants required
            # depending on the sample token type
            match token['type']:
                case 'bin_op':
                    num_consts_required += 1
                case 'const':
                    num_consts_required -= 1

            token['forced_const'] = False if pre_softmax_mask is None else True

            tokens.append(token)
            i += 1

        return Equation(tokens)

    # Calculate probabilities of a set of equations, z, under q
    def pdf(self, z, token_set):

        probabilities = []

        for eq in z:

            #print(eq.get_infix())

            self._net.reset(1)

            x = torch.zeros(self._net.num_inputs())

            prob = 1.0
            for t in eq.tokens():

                # Apply mask to force pdf to only be over const tokens
                pre_softmax_mask = \
                    self._consts_mask if t['forced_const'] else None

                x = self._net.forward(x, pre_softmax_mask)
                #print('PDF:', x)

                # Generate one hot vector for current token
                one_hot = torch.zeros(self._net.num_inputs())
                one_hot[t['id']] = 1.0

                prob = prob * torch.sum(x * one_hot)

                #print(prob)

            probabilities.append(prob)

        return torch.stack(probabilities)


class NN(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size):
        super().__init__()

        '''
        self._hidden_size = hidden_size

        self._l1 = torch.nn.GRUCell(num_inputs, self._hidden_size)
        self._l2 = torch.nn.Linear(self._hidden_size, num_outputs)

        self._num_inputs = num_inputs

        self._l1 = torch.nn.Linear(self._num_inputs, num_outputs)
        '''

        self._num_inputs = num_inputs

        #self._bias = torch.nn.Parameter(torch.randn(num_outputs))
        #self._bias = torch.nn.Parameter(torch.randn(1))
        # More 1.0s
        self._bias = torch.nn.Parameter(torch.Tensor([2.0]))
        #self._bias = torch.nn.Parameter(torch.Tensor([100.0]))
        # More x_0s
        #self._bias = torch.nn.Parameter(torch.Tensor([-2.0]))
        #self._bias = torch.nn.Parameter(torch.Tensor([-100.0]))

    def forward(self, x, pre_softmax_mask=None):

        #print(self._bias)
        x = torch.nn.functional.sigmoid(self._bias)
        #print(x)
        x = torch.ones(2) * x
        x = torch.Tensor([1.0, -1.0]) * x
        x = torch.Tensor([0.0, 1.0]) + x
        #print(x)
        return x

        '''
        x = torch.nn.functional.softmax(self._bias)
        return x
        '''


        '''
        x = self._l1(x)
        x = torch.nn.functional.softmax(x)
        return x
        '''


        '''
        # Check hidden state has been initialised
        if not hasattr(self, '_hx'):
            raise RuntimeError('Must call reset() before forward()')

        # GRU layer
        x = self._l1(x, self._hx)
        self._hx = x

        # Linear layer
        x = self._l2(x)

        # Apply binary mask before the softmax - this is equivalent to
        # preventing some of the tokens being sampled
        # TODO: I do not know whether this should be in-place
        if pre_softmax_mask is not None:
            x += pre_softmax_mask

        # Softmax layer
        x = torch.nn.functional.softmax(x)
        '''

        return x

    def reset(self, batch_size):
        pass
        #self._hx = torch.zeros(self._hidden_size)

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

        eq = copy.deepcopy(self._eq)

        # Convert consts to list of relevant data size
        for token in eq:
            if token['type'] == 'const' and not isinstance(token['op'], str):
                token['op'] = np.array([token['op']] * len(x))

        # Substitute variables for data, x
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

    # Return infix string
    def get_infix(self):

        eq = copy.deepcopy(self._eq)
        eq.reverse()

        stack = []

        for t in eq:

            # If token is a constant, push onto stack
            if t['type'] == 'const':
                stack.append(str(t['op']))

            # Otherwise print operators with elements from stack
            else:

                if t['type'] == 'bin_op':
                    stack.append('(' + stack.pop() + ' ' + t['op']
                                 + ' ' + stack.pop() + ')')
                elif t['type'] == 'un_op':
                    stack.append(t['op'] + '(' + stack.pop() + ')')
                else:
                    raise RuntimeError(
                            t['type'] + ' is not a recognised token type')

        return stack.pop()

    def num_tokens(self):
        return len(self._eq)

    def tokens(self):
        return self._eq

    def __repr__(self):
        return str(self._eq)
