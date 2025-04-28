import copy
import numpy as np
import scipy
import re


# A class for representing analytic equations and evaluating them
class Equation:

    # Tokens should be input in polish notation
    def __init__(self, tokens=None, infix_str=None, token_set=None):

        if tokens is not None:

            # Token given in polish notation
            self._eq = tokens

        # Written expression can be given in infix notation
        else:

            polish_str = infix_to_polish(infix_str, token_set)
            token_strs = polish_str.split()

            self._eq = []

            for t_str in token_strs:
                token = next((
                    copy.deepcopy(token) for token in token_set
                    if token['op'] == t_str), None)
                self._eq.append(token)

        # Check number of opt_consts
        self._num_opt_consts = sum(1 for t in self._eq
                                   if t['op'] == 'opt_const')

        # Calculate number of distr_consts
        self._num_distr_consts = sum(1 for t in self._eq
                                     if t['op'] == 'distr_const')

    # Evaluate equation according to data variable values, x.
    # Returns None if equation cannot be evaluated, for example, if there is
    # a divide by 0, etc.
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

                if t['op'] == '*':
                    stack.append(stack.pop() * stack.pop())

                elif t['op'] == '/':

                    numerator = stack.pop()
                    denominator = stack.pop()

                    # Check for divide by 0
                    if np.any(denominator == 0):
                        return None

                    stack.append(numerator / denominator)

                elif t['op'] == '+':
                    stack.append(stack.pop() + stack.pop())

                elif t['op'] == '-':
                    stack.append(stack.pop() - stack.pop())

                elif t['op'] == 'cos':
                    stack.append(np.cos(stack.pop()))

                elif t['op'] == 'sin':
                    stack.append(np.sin(stack.pop()))

                elif t['op'] == 'exp':
                    stack.append(np.exp(stack.pop()))

                elif t['op'] == 'log':
                    stack.append(np.log(stack.pop()))

                else:
                    raise RuntimeError(
                        t['op'] + ' is not a recognised operator'
                    )

        return stack.pop()

    # Return infix string
    def get_infix(self, simplify=False):

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

        eq_str = stack.pop()

        if simplify:
            eq_str = self._simplify(eq_str)

        return eq_str

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
    def apply_pre_softmax_mask(self, max_num_tokens, net_masks):

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
                token['pre_softmax_mask'] = copy.deepcopy(
                    net_masks.compose_mask(['consts'])
                )
            elif i + num_consts_required + 1 >= max_num_tokens:
                token['pre_softmax_mask'] = copy.deepcopy(
                    net_masks.compose_mask(['un_ops', 'consts'])
                )

            # Increase or decrease the number of constants required
            # depending on the sample token type
            if token['type'] == 'bin_op':
                num_consts_required += 1
            elif token['type'] == 'const':
                num_consts_required -= 1

    def convert_distr_to_opt_consts(self):
        for t in self._eq:
            t['op'] = 'opt_const' if t['op'] == 'distr_const' else t['op']
        self._num_opt_consts = sum(1 for t in self._eq
                                   if t['op'] == 'opt_const')

    def to_json(self):

        j = {
            'eq': self.get_infix(),
            'eq (simplified)': self.get_infix(True)
        }

        return j

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

    # Use sympy to simplify equation and return simplified string
    def _simplify(self, eq_str):

        from sympy import sympify, expand
        expr = sympify(eq_str)
        expr = expand(expr)
        return str(expr)

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


def infix_to_polish(infix, token_set):

    # Operator precedence and associativity
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    precedence.update({t['op']: 4 for t in token_set if t['type'] == 'un_op'})

    associativity = {'+': 'L', '-': 'L', '*': 'L', '/': 'L', '^': 'R'}
    associativity.update({t['op']: 'R' for t in token_set if t['type'] == 'un_op'})

    is_unary = {'+': False, '-': False, '*': False, '/': False, '^': False}
    is_unary.update({t['op']: True for t in token_set if t['type'] == 'un_op'})

    tokens = tokenize(infix)  # Use the new tokenizer

    # Check all tokens are in token set
    for t_str in tokens:
        if t_str == '(' or t_str == ')':
            continue
        found = False
        for t in token_set:
            if t_str == t['op']:
                found = True
                break
        if not found:
            raise ValueError(f'{t_str} is not in the token set')

    op_stack = []
    operand_stack = []

    for token in tokens:
        if token == '(':
            op_stack.append(token)
        elif token == ')':
            while op_stack and op_stack[-1] != '(':
                op = op_stack.pop()
                if is_unary.get(op, False):
                    operand = operand_stack.pop()
                    operand_stack.append(f"{op} {operand}")
                else:
                    operand2 = operand_stack.pop()
                    operand1 = operand_stack.pop()
                    operand_stack.append(f"{op} {operand1} {operand2}")
            op_stack.pop()  # Remove '('
            if op_stack and is_unary.get(op_stack[-1], False):
                func = op_stack.pop()
                operand = operand_stack.pop()
                operand_stack.append(f"{func} {operand}")
        elif token in precedence:
            while (op_stack and op_stack[-1] != '(' and
                   ((associativity[token] == 'L' and precedence[token] <= precedence.get(op_stack[-1], 0)) or
                    (associativity[token] == 'R' and precedence[token] < precedence.get(op_stack[-1], 0)))):
                op = op_stack.pop()
                if is_unary.get(op, False):
                    operand = operand_stack.pop()
                    operand_stack.append(f"{op} {operand}")
                else:
                    operand2 = operand_stack.pop()
                    operand1 = operand_stack.pop()
                    operand_stack.append(f"{op} {operand1} {operand2}")
            op_stack.append(token)
        else:
            operand_stack.append(token)  # Operands like x_0, 3.14

    while op_stack:
        op = op_stack.pop()
        if is_unary.get(op, False):
            operand = operand_stack.pop()
            operand_stack.append(f"{op} {operand}")
        else:
            operand2 = operand_stack.pop()
            operand1 = operand_stack.pop()
            operand_stack.append(f"{op} {operand1} {operand2}")

    return operand_stack[0]


def tokenize(expression):
    # Pattern matches identifiers, numbers, operators, and parentheses
    token_pattern = r'([a-zA-Z][a-zA-Z0-9_]*|\d+\.\d+|\d+|[+\-*/^()])'
    tokens = re.findall(token_pattern, expression)
    return tokens
