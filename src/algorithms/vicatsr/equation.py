import copy
import numpy as np
import scipy


# A class for representing analytic equations and evaluating them
class Equation:

    # Tokens should be input in polish notation
    def __init__(self, tokens):

        # Equation is represented in polish notation
        self._eq = tokens

        # Check number of opt_consts
        self._num_opt_consts = sum(1 for t in tokens
                                   if t['op'] == 'opt_const')

        # Calculate number of distr_consts
        self._num_distr_consts = sum(1 for t in tokens
                                     if t['op'] == 'distr_const')

    # Evaluate equation according to data variable values, x.
    def evaluate(self, x):

        eq = copy.deepcopy(self._eq)

        # Replace opt and distr consts with values
        eq = self._replace_opt_consts(eq)
        eq = self._replace_distr_consts(eq)

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

        # Replace opt and distr consts with values
        eq = self._replace_opt_consts(eq)
        eq = self._replace_distr_consts(eq)

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

    def distr_const_tokens(self):
        return [t for t in self._eq if t['op'] == 'distr_const']

    def num_opt_consts(self):
        return self._num_opt_consts

    def num_distr_consts(self):
        return self._num_distr_consts

    def num_float_consts(self):
        return sum(1 for e in self._eq if e['sub_type'] == 'float_const')

    def set_opt_consts(self, x):

        if len(x) != self._num_opt_consts:
            raise ValueError(
                f"Expects {self._num_opt_consts} opt consts but "
                f"{len(x)} was given"
            )

        i = 0
        for token in self._eq:
            if token['op'] == 'opt_const':
                token['value'] = x[i]
                i += 1

    def set_distr_consts(self, x):

        if len(x) != self._num_distr_consts:
            raise ValueError(
                f"Expects {self._num_distr_consts} distr consts but "
                f"{len(x)} was given"
            )

        i = 0
        for token in self._eq:
            if token['op'] == 'distr_const':
                token['value'] = x[i]
                i += 1

    # If masks have not been calculated, do that here
    def apply_pre_softmax_mask(self, max_num_tokens,
                               consts_mask, un_ops_consts_mask):

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

    def convert_distr_to_opt_consts(self):
        for t in self._eq:
            t['op'] = 'opt_const' if t['op'] == 'distr_const' else t['op']
        self._num_opt_consts = sum(1 for t in self._eq
                                   if t['op'] == 'opt_const')

    # Replace opt_const with values
    def _replace_opt_consts(self, eq):

        if self._num_opt_consts != 0:
            i = 0
            for token in eq:
                if token['op'] == 'opt_const':
                    if token['value'] is not None:
                        token['op'] = token['value']
                        i += 1
                    else:
                        raise ValueError(
                            'Trying to evaluate an equation that has opt '
                            'const tokens but no opt const values'
                        )

        return eq

    # Replace distr_const with values
    def _replace_distr_consts(self, eq):

        if self._num_distr_consts != 0:
            i = 0
            for token in eq:
                if token['op'] == 'distr_const':
                    if token['value'] is not None:
                        token['op'] = token['value']
                        i += 1
                    else:
                        raise ValueError(
                            'Trying to evaluate an equation that has distr '
                            'const tokens but no distr const values'
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
