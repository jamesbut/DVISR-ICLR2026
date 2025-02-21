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
import math
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

        # Add constants as tokens
        if 'consts' in config['operators']:
            for c in config['operators']['consts']:
                self._token_set.append({"op": c, "type": "const",
                                        "id": self._token_id})
                self._token_id += 1

        # Number of equations sampled to calculate expected loss
        self._num_eq_samples = config['num_eq_samples']

        # Maximum equation tree depth
        self._max_depth = None
        if 'max_depth' in config:
            # self._max_depth = config['max_depth']
            raise NotImplementedError('Max tree depth not yet implemented')

        # Maximum number of tokens in generated equations
        self._max_num_tokens = config['max_num_tokens']

        # Learning rate for optimiser
        self._lr = config['learning_rate']

        # Number of training steps
        self._num_steps = config['num_steps']

        # Size of RNN hidden layer
        self._hidden_layer_size = config['rnn_hidden_layer_size']

        # Seed random number generators
        self._seed = config.get('seed', None)
        if self._seed is not None:
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)

    def train(self, data):

        self._initialise(data)

        # self._maximise_likelihood(data)
        self._maximise_elbo(data)

    def _maximise_likelihood(self, data):

        optimiser = torch.optim.RMSprop(self._q._net.parameters(), lr=self._lr)

        for i in range(self._num_steps):

            # Sample z from surrogate q
            sampled_z = [self._q.sample_and_optimise(data, log_likelihood)
                         for i in range(self._num_eq_samples)]

            # Optimise equation constants if required
            sampled_z = [optimise_eq_consts(z, data, log_likelihood)
                         for z in sampled_z]

            # Calculate likelihoods of sampled models
            log_likelihoods = torch.tensor(
                [log_likelihood(data, z) for z in sampled_z],
                requires_grad=False
            )

            '''
            all_exps = enumerate_expressions(self._token_set, self._max_num_tokens)
            for e in all_exps:
                print(e.get_infix())
            exit()
            '''

            '''
            for z, l in zip(sampled_z, log_likelihoods):
                print('z: ' + z.get_infix() + '    likelihood: ' + str(l))
            exit()
            '''

            rewards = log_likelihoods
            baseline = rewards.mean()
            rewards = rewards - baseline

            loss = torch.stack(
                [-self._q.log_pdf(z) * r for z, r in zip(sampled_z, rewards)]
            ).mean()

            print('Step: {}   Loss: {}'.format(str(i), loss.item()))

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()

        sampled_z = [self._q.sample_and_optimise(data, log_likelihood)
                     for i in range(self._num_eq_samples)]
        for z in sampled_z:
            print('z: ' + z.get_infix() + '    pdf: '
                  + str(self._q.pdf(z).item()))

    def _maximise_elbo(self, data):

        optimiser = torch.optim.RMSprop(self._q._net.parameters(), lr=self._lr)

        for i in range(self._num_steps):

            # Sample z from surrogate q
            sampled_z = [self._q.sample_and_optimise(data, log_likelihood)
                         for i in range(self._num_eq_samples)]

            # Calculate log likelihoods of sampled models
            log_likelihoods = torch.tensor(
                [log_likelihood(data, z) for z in sampled_z],
                requires_grad=False
            )

            # Calculate log q(z) under the surrogate distribution for samples
            # models
            # NOTE: This .detach() makes a big difference to optimisation
            log_q_zs = torch.stack([self._q.log_pdf(z) for z in sampled_z]).detach()

            # Calculate priors, ln p(z), for sampled models
            log_priors = torch.tensor(
                [self._log_prior(z) for z in sampled_z],
                requires_grad=False
            )

            # Calculate ELBO
            elbos = log_likelihoods + log_priors - log_q_zs

            rewards = elbos

            baseline = rewards.mean()
            rewards = rewards - baseline

            loss = torch.stack(
                [-self._q.log_pdf(z) * r for z, r in zip(sampled_z, rewards)]
            ).mean()

            print('Step: {}   Loss: {}'.format(str(i), loss.item()))

            optimiser.zero_grad()

            loss.backward()

            optimiser.step()

        '''
        sampled_z = [self._q.sample_and_optimise(data, log_likelihood)
                     for i in range(self._num_eq_samples)]
        for z in sampled_z:
            print('z: ' + z.get_infix() + '    pdf: ' + str(self._q.pdf(z).item()))
        '''

        true_posterior, all_exps = self._true_posterior(data)

        for p_z_x, z in zip(true_posterior, all_exps):
            print(
                'z: ' + z.get_infix() + '    q(z): '
                + str(self._q.pdf(z).item()) + '    p(z|x): '
                + str(p_z_x)
            )

    def _initialise(self, data):

        # Finish creating token set
        for i in range(len(data['x'][0])):
            self._token_set.append({"op": "x_" + str(i), "type": "const",
                                    "id": self._token_id})
            self._token_id += 1

        # Calculate total number of models
        self._total_num_eqs = calculate_total_num_eqs(self._token_set,
                                                      self._max_num_tokens)

        # Create surrogate distribution, q, which is optimised to approximate
        # the posterior
        self._q = q(self._token_set, self._hidden_layer_size,
                    self._max_depth, self._max_num_tokens)

    def _prior(self, z):

        # Calculate uniform prior for now
        total_num_eqs = calculate_total_num_eqs(self._token_set,
                                                self._max_num_tokens)
        prior = 1 / total_num_eqs
        return prior

    def _log_prior(self, z):
        return math.log(self._prior(z))

    # Calculate the true posterior for all enumerated models
    def _true_posterior(self, data):

        # Enumerate all expressions
        all_exps = enumerate_expressions(self._token_set, self._max_num_tokens)

        # Calculate p(x) based on the law of total probability
        p_x = sum([likelihood(data, z) * self._prior(z) for z in all_exps])

        # Calculate p(z|x) for all expressions
        p_z_x = [likelihood(data, z) * self._prior(z) / p_x for z in all_exps]

        '''
        for z, posterior in zip(all_exps, p_z_x):
            print('z: ' + z.get_infix() + '    posterior: ' + str(posterior))
        '''

        return p_z_x, all_exps


def log_likelihood(data, z):

    likelihoods = []
    means = z.evaluate(data['x'])
    for i in range(len(means)):
        likelihoods.append(scipy.stats.norm.logpdf(data['y'][i],
                                                   loc=means[i],
                                                   scale=1.0))
    return sum(likelihoods)


def likelihood(data, z):

    likelihoods = []
    means = z.evaluate(data['x'])
    for i in range(len(means)):
        likelihoods.append(scipy.stats.norm.pdf(data['y'][i],
                                                loc=means[i],
                                                scale=1.0))
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


# Enumerate all expressions according to a specific token set and a maximum
def enumerate_expressions(token_set, max_num_tokens):

    l_m = max_num_tokens

    # Split tokens by type
    consts = [t for t in token_set if t['type'] == 'const']
    un_ops = [t for t in token_set if t['type'] == 'un_op']
    bin_ops = [t for t in token_set if t['type'] == 'bin_op']

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
        e.apply_pre_softmax_mask(max_num_tokens)

    return all_expressions


# Surrogate distribution, q, which is optimised to approximate the
# posterior.
# It currently consists of a recurrent neural network that outputs
# a sequence of categorical distribution parameters.
class q:

    def __init__(self, token_set, hidden_layer_size, max_depth, max_num_tokens):

        # Create recurrent neural network
        self._net = NN(len(token_set), len(token_set), hidden_layer_size)

        self._max_depth = max_depth
        self._max_num_tokens = max_num_tokens
        self._token_set = token_set

        # A mask to apply so that only constants are sampled
        global consts_mask
        consts_mask = [0.0 if t['type'] == 'const' else -1e9
                       for t in self._token_set]
        consts_mask = torch.from_numpy(np.array(consts_mask))

        # A mask to apply so that only unary operators and consts are sampled
        global un_ops_consts_mask
        un_ops_consts_mask = [0.0 if t['type'] == 'un_op'
                                     or t['type'] == 'const' else -1e9
                              for t in self._token_set]
        un_ops_consts_mask = \
            torch.from_numpy(np.array(un_ops_consts_mask))

    def sample(self):

        self._net.reset(1)

        # Loop until max depth or sufficient number of constants have been
        # sampled
        tokens = []
        x = torch.zeros(self._net.num_inputs())
        num_consts_required = 1

        while num_consts_required > 0:

            pre_softmax_mask = None
            # Apply mask to only sample unary operators and constants
            if self._max_num_tokens - len(tokens) <= num_consts_required + 1:
                pre_softmax_mask = un_ops_consts_mask

            # Apply mask to only sample constants
            if self._max_num_tokens - len(tokens) <= num_consts_required:
                pre_softmax_mask = consts_mask

            # Pass input through network
            out = self._net.forward(x, pre_softmax_mask).detach().numpy()

            # Sample token from categorical distribution
            token = copy.deepcopy(np.random.choice(self._token_set, 1, p=out)[0])

            # Generate next network input
            x = torch.zeros_like(x)
            x[token['id']] = 1.0

            # Increase or decrease the number of constants required
            # depending on the sample token type
            match token['type']:
                case 'bin_op':
                    num_consts_required += 1
                case 'const':
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

        x = torch.zeros(self._net.num_inputs())

        prob = 1.0
        for t in z.tokens():

            out = self._net.forward(x, t['pre_softmax_mask'])

            # Generate one hot vector for current token
            one_hot = torch.zeros(self._net.num_inputs())
            one_hot[t['id']] = 1.0

            prob *= torch.sum(out * one_hot)

            # Set next network input
            x = one_hot.clone().detach()

        return prob

    # Calculate log probability of an equation, z, under q
    def log_pdf(self, z):

        self._net.reset(1)

        x = torch.zeros(self._net.num_inputs())

        log_prob = 0.0
        for t in z.tokens():

            x = self._net.forward(x, t['pre_softmax_mask'])

            # Generate one hot vector for current token
            one_hot = torch.zeros(self._net.num_inputs())
            one_hot[t['id']] = 1.0

            log_prob += torch.log(torch.sum(x * one_hot))

            # Set next network input
            x = one_hot.clone().detach()

        return log_prob


class NN(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size):
        super().__init__()

        self._hidden_size = hidden_size

        if hidden_size == 0:
            self._l1 = None
            self._l2 = torch.nn.Linear(num_inputs, num_outputs)
        else:
            self._l1 = torch.nn.GRUCell(num_inputs, self._hidden_size)
            self._l2 = torch.nn.Linear(self._hidden_size, num_outputs)

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

    def forward(self, x, pre_softmax_mask=None):

        # Check hidden state has been initialised
        if not hasattr(self, '_hx'):
            raise RuntimeError('Must call reset() before forward()')

        # GRU layer
        if self._l1 is not None:
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

        return x

    def reset(self, batch_size):
        self._hx = torch.zeros(self._hidden_size)

    def num_inputs(self):
        return self._num_inputs

    def num_outputs(self):
        return self._num_outputs


# A class for representing analytic equations and evaluating them
class Equation:

    # Tokens should be input in polish notation
    def __init__(self, tokens):

        # Equation is represented in polish notation
        self._eq = tokens

        # Check whether equation has any consts to optimise
        self._num_opt_consts = 0
        for t in tokens:
            if t['op'] == 'opt_const':
                self._num_opt_consts += 1

        # Optimisable constant values
        self._opt_consts = None

    # Evaluate equation according to data variable values, x.
    def evaluate(self, x):

        eq = copy.deepcopy(self._eq)

        # Replace opt consts with values
        eq = self._replace_opt_consts(eq)

        # Convert consts to list of relevant data size
        for token in eq:
            if token['type'] == 'const' and not isinstance(token['op'], str):
                token['op'] = np.array([token['op']] * len(x))

        # Substitute variables for data, x
        for token in eq:
            for i in range(x.shape[1]):
                if not isinstance(token['op'], np.ndarray):
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

        # Replace opt consts with values
        eq = self._replace_opt_consts(eq)

        eq.reverse()

        stack = []

        for t in eq:

            # If token is a constant, push onto stack
            if t['type'] == 'const':
                if isinstance(t['op'], str):
                    stack.append(t['op'])
                else:
                    stack.append("{:.4f}".format(t['op']))

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

    def num_opt_consts(self):
        return self._num_opt_consts

    def set_opt_consts(self, x):

        if len(x) != self._num_opt_consts:
            raise ValueError(
                f"Expects {self._num_opt_consts} opt consts but "
                f"{len(x)} was given"
            )

        self._opt_consts = x

    # If masks have not been calculated, do that here
    def apply_pre_softmax_mask(self, max_num_tokens):

        # Check whether forced consts have already been applied
        for token in self._eq:
            if 'pre_softmax_mask' in token:
                raise RuntimeError(
                    'Trying to apply pre softmax masks to an equation that '
                    'already has them set'
                )

        num_consts_required = 1
        for i, token in enumerate(self._eq):

            # Determine whether and which mask would have been used
            token['pre_softmax_mask'] = None
            if i + num_consts_required >= max_num_tokens:
                token['pre_softmax_mask'] = copy.deepcopy(consts_mask)
            elif i + num_consts_required + 1 >= max_num_tokens:
                token['pre_softmax_mask'] = copy.deepcopy(un_ops_consts_mask)

            # Increase or decrease the number of constants required
            # depending on the sample token type
            match token['type']:
                case 'bin_op':
                    num_consts_required += 1
                case 'const':
                    num_consts_required -= 1

    # Replace opt_const with values
    def _replace_opt_consts(self, eq):

        if self._num_opt_consts != 0:
            if self._opt_consts is not None:
                i = 0
                for token in eq:
                    if token['op'] == 'opt_const':
                        token['op'] = self._opt_consts[i]
                        i += 1
            else:
                raise ValueError(
                    'Trying to evaluate an equation that has opt const tokens '
                    'but no opt const values'
                )

        return eq

    def __repr__(self):
        return str(self._eq)


# Optimise consts in equation to maximise log likelihood
def optimise_eq_consts(eq, data, log_likelihood_func):

    # If there are no consts to optimise just return original equation
    if eq.num_opt_consts() == 0:
        return eq

    # Initial guess of all ones
    init_x = np.ones(eq.num_opt_consts())

    def min_func(x, eq, data, log_likelihood_func):

        # Evaluate equation with opt consts set as x
        eq.set_opt_consts(x)

        log_likelihood = log_likelihood_func(data, eq)

        return -log_likelihood

    # Optimise log likelihood with respect to op constants
    res = scipy.optimize.minimize(min_func, init_x,
                                  args=(eq, data, log_likelihood_func),
                                  method='bfgs')

    if not res['success']:
        raise RuntimeError('Scipy minimize failed')

    eq.set_opt_consts(res['x'])

    return eq
