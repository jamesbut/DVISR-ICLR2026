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
import math
import itertools
import matplotlib.pyplot as plt
from equation import Equation


class VICatSR(Algorithm):

    def __init__(self, config):

        # Prepare binary and unary operations as tokens
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
        self._lr = config['learning_rate']

        # Number of training steps
        self._num_steps = config['num_steps']

        # Size of RNN hidden layer
        self._hidden_layer_size = config['rnn_hidden_layer_size']

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

        # Evidence only needs to be computed once
        self._evidence = None

        # Seed random number generators
        self._seed = config.get('seed', None)
        if self._seed is not None:
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)

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

        for i in range(self._num_steps):

            # Sample z from surrogate q
            sampled_z = [self._q.sample_and_optimise(data, log_likelihood)
                         for i in range(self._num_eq_samples)]

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

        '''
        sampled_z = [self._q.sample_and_optimise(data, log_likelihood)
                     for i in range(self._num_eq_samples)]
        for z in sampled_z:
            print('z: ' + z.get_infix() + '    pdf: '
                  + str(self._q.pdf(z).item()))
        '''

        all_exps = self._enumerate_expressions(data)
        for z in all_exps:
            print('z: ' + z.get_infix() + '    q(z): '
                  + str(self._q.pdf(z).item()) + '     p(x|z): '
                  + str(likelihood(data, z)))

        return self._q, all_exps

    def _maximise_elbo(self, data):

        optimiser = torch.optim.RMSprop(self._q._net.parameters(), lr=self._lr)

        mus = []
        kl_divs = []
        for i in range(self._num_steps):

            # Calculate ELBO
            elbos, sampled_z = self.elbos(data, self._num_eq_samples)

            if self._distr_over_consts:
                mu = self._q.net_outs(sampled_z[0])[-1][-2]
                mus.append(mu)

            if self._track_kl_divergence:
                kl_divergence = self.kl_divergence(data, num_samples=100)
                kl_divs.append(kl_divergence)

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

        if self._plotting:
            plt.plot(range(self._num_steps), mus)
            plt.show()
            if self._track_kl_divergence:
                plt.plot(range(self._num_steps), kl_divs)
                plt.show()

        '''
        sampled_z = [self._q.sample_and_optimise(data, log_likelihood)
                     for i in range(self._num_eq_samples)]
        for z in sampled_z:
            print('z: ' + z.get_infix() + '    pdf: ' + str(self._q.pdf(z)[0].item()))
        '''

        true_posteriors, all_exps = self.posteriors(data)

        for p_z_x, z in zip(true_posteriors, all_exps):
            print(
                'z: ' + z.get_infix() + '    q(z): '
                + str(self._q.pdf(z).item()) + '    p(z|x): '
                + str(p_z_x)
            )
            consts_params = self._q.get_consts_params(z)
            print('     q consts params:', consts_params)

        kl_divergence = self.kl_divergence(data, num_samples=1000)
        print('KL divergence:', kl_divergence)
        print('------------------------------')

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
                    self._max_depth, self._max_num_tokens,
                    self._distr_over_consts, self._q_const_variance,
                    self._consts_mask, self._un_ops_consts_mask)

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
    def elbos(self, data, num_samples):

        # Sample z from surrogate q
        sampled_z = [self._q.sample_and_optimise(data, log_likelihood)
                     for i in range(num_samples)]

        # Calculate log likelihoods of sampled models
        log_likelihoods = torch.tensor(
            [log_likelihood(data, z) for z in sampled_z],
            requires_grad=False
        )

        # Calculate log q(z) under the surrogate distribution for samples
        # models
        log_q_zs = torch.stack(
            [self._q.log_pdf(z) for z in sampled_z]
        ).detach()

        # Calculate priors, ln p(z), for sampled models
        log_priors = torch.tensor(
            [self._log_prior(z) for z in sampled_z],
            requires_grad=False
        )

        # Calculate ELBO
        return log_likelihoods + log_priors - log_q_zs, sampled_z

    # Calculate the KL divergence between q(z) and p(z|x)
    def kl_divergence(self, data, num_samples):

        elbo = self.elbos(data, num_samples)[0].mean()

        # Enumerate all expressions
        all_z = self._enumerate_expressions(data)

        # Calculate KL divergence
        kl_divergence = self.log_evidence(data, all_z) - elbo

        return kl_divergence.item()

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
            raise RuntimeError('Cannot plot distributions for more than y=c')

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


# Surrogate distribution, q, which is optimised to approximate the
# posterior.
# It currently consists of a recurrent neural network that outputs
# a sequence of categorical distribution parameters.
class q:

    def __init__(self, token_set, hidden_layer_size, max_depth,
                 max_num_tokens, distr_over_consts, const_variance,
                 consts_mask, un_ops_consts_mask):

        # Create recurrent neural network
        self._net = NN(len(token_set), len(token_set),
                       hidden_layer_size, distr_over_consts,
                       True if const_variance is None else False)

        self._max_depth = max_depth
        self._max_num_tokens = max_num_tokens
        self._token_set = token_set
        self._distr_over_consts = distr_over_consts

        self._consts_mask = consts_mask
        self._un_ops_consts_mask = un_ops_consts_mask

        # Variance for normal distribution over constants
        # If set to None, this is also optimised
        self._const_variance = const_variance

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
                pre_softmax_mask = self._un_ops_consts_mask

            # Apply mask to only sample constants
            if self._max_num_tokens - len(tokens) <= num_consts_required:
                pre_softmax_mask = self._consts_mask

            # Pass input through network
            out = self._net.forward(x, pre_softmax_mask).detach().numpy()

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

            # Generate next network input
            x = torch.zeros_like(x)
            x[token['id']] = 1.0

            # TODO: Might have to input sampled value for constants back into
            # the network

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

        eq = self.sample()

        # Do not optimise if using distribution over constants
        if self._distr_over_consts:
            return eq
        else:
            return optimise_eq_consts(eq, data, log_likelihood_func)

    # Calculate probability of an equation, z, under q
    def pdf(self, z):

        self._net.reset(1)

        x = torch.zeros(self._net.num_inputs())

        probs = []
        for t in z.tokens():

            out = self._net.forward(x, t['pre_softmax_mask'])

            # TODO: I might not have to generate a one hot encoding and
            # multiply, I might just be able to index the tensor?
            # Generate one hot vector for current token
            one_hot = torch.zeros(self._net.num_inputs())
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

            # Set next network input
            x = one_hot.clone().detach()

        return math.prod(probs)

    # Calculate log probability of an equation, z, under q
    def log_pdf(self, z):

        self._net.reset(1)

        x = torch.zeros(self._net.num_inputs())

        log_probs = []
        for t in z.tokens():

            out = self._net.forward(x, t['pre_softmax_mask'])

            # Generate one hot vector for current token
            one_hot = torch.zeros(self._net.num_inputs())
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

            # Set next network input
            x = one_hot.clone().detach()

        return sum(log_probs)

    # Get all network outputs for a particular equation
    def net_outs(self, z):

        self._net.reset(1)

        x = torch.zeros(self._net.num_inputs())

        net_outs = []
        for t in z.tokens():

            out = self._net.forward(x, t['pre_softmax_mask'])
            net_outs.append(out)

            # TODO: I might not have to generate a one hot encoding and
            # multiply, I might just be able to index the tensor?
            # Generate one hot vector for current token
            one_hot = torch.zeros(self._net.num_inputs())
            one_hot[t['id']] = 1.0

            # Set next network input
            x = one_hot.clone().detach()

        return torch.stack(net_outs).detach().numpy()

    # Get means and variances output by the network for all constants in
    # an equation
    def get_consts_params(self, z):
        return [[out[-2], out[-1]]
                for out, token in zip(self.net_outs(z), z.tokens())
                if token['op'] == 'distr_const']


class NN(torch.nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size,
                 distr_over_consts: bool, const_variance: bool):
        super().__init__()

        self._hidden_size = hidden_size

        self._l1 = None
        self._consts_layer = None

        if hidden_size == 0:

            self._l2 = torch.nn.Linear(num_inputs, num_outputs)

            if distr_over_consts:
                self._consts_layer = torch.nn.Linear(num_inputs, 2)

        else:
            self._l1 = torch.nn.GRUCell(num_inputs, self._hidden_size)
            self._l2 = torch.nn.Linear(self._hidden_size, num_outputs)

            if distr_over_consts:
                self._consts_layer = torch.nn.Linear(self._hidden_size, 2)

        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

        self._const_variance = const_variance

    def forward(self, x, pre_softmax_mask=None):

        # Check hidden state has been initialised
        if not hasattr(self, '_hx'):
            raise RuntimeError('Must call reset() before forward()')

        # GRU layer
        if self._l1 is not None:
            x = self._l1(x, self._hx)
            self._hx = x

        # Linear layer that produces logits for the categorical distribution
        cat_logits = self._l2(x)

        # Apply binary mask before the softmax - this is equivalent to
        # preventing some of the tokens being sampled
        # TODO: I do not know whether this should be in-place
        if pre_softmax_mask is not None:
            cat_logits += pre_softmax_mask

        # Softmax layer that converts logits to probabilities
        cat_params = torch.nn.functional.softmax(cat_logits, dim=0)
        output = cat_params

        # Output parameter of constant distribution
        if self._consts_layer is not None:
            const_out = self._consts_layer(x)

            if self._const_variance:
                const_out = torch.where(
                    torch.tensor([False, True]),
                    torch.nn.functional.softplus(const_out),
                    const_out
                )

            output = torch.cat((output, const_out), dim=0)

        return output

    def reset(self, batch_size):
        self._hx = torch.zeros(self._hidden_size)

    def num_inputs(self):
        return self._num_inputs

    def num_outputs(self):
        return self._num_outputs


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
