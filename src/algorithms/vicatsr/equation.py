import copy
import numpy as np
import scipy


# A class for representing analytic equations and evaluating them
class Equation:

    # Tokens should be input in polish notation
    def __init__(self, tokens=None, infix_str=None, token_set=None):

        if tokens is not None:

            # Token given in polish notation
            self._eq = tokens

        # Written expression can be given in infix notation
        else:

            polish_str = infix_to_polish(infix_str)
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


def infix_to_polish(infix):
    """
    Convert an infix equation string to Polish (prefix) notation.

    Parameters:
    - infix: String representing an infix equation (e.g., "a + b * c")

    Returns:
    - String in Polish notation (e.g., "+ a * b c")
    """
    # Operator precedence
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    # Unary operator flag (e.g., "-a" at start or after operator)
    unary_ops = {'-'}  # Could add '+' for unary plus if needed

    # Tokenize the infix expression
    tokens = tokenize_infix(infix)
    if not tokens:
        return ""

    output = []
    operator_stack = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Operand (variable or constant)
        if token.isalnum() or '_' in token:
            output.append(token)

        # Left parenthesis
        elif token == '(':
            operator_stack.append(token)

        # Right parenthesis
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.insert(0, operator_stack.pop())
            if operator_stack and operator_stack[-1] == '(':
                operator_stack.pop()  # Discard '('
            else:
                raise ValueError("Mismatched parentheses")

        # Operator
        elif token in precedence or token in unary_ops:
            # Check if unary (at start or after another operator/parenthesis)
            is_unary = (i == 0) or (i > 0 and (tokens[i-1] in precedence or tokens[i-1] in unary_ops or tokens[i-1] == '('))

            if is_unary and token in unary_ops:
                # Treat as unary: push and wait for operand
                operator_stack.append('u' + token)  # Prefix with 'u' to distinguish
            else:
                # Binary operator
                while (operator_stack and operator_stack[-1] not in '(' and
                       (operator_stack[-1].startswith('u') or
                        precedence.get(operator_stack[-1], 0) >= precedence.get(token, 0))):
                    output.insert(0, operator_stack.pop().lstrip('u'))
                operator_stack.append(token)

        i += 1

    # Pop remaining operators
    while operator_stack:
        op = operator_stack.pop()
        if op == '(':
            raise ValueError("Mismatched parentheses")
        output.insert(0, op.lstrip('u'))

    # Return as string
    return " ".join(output)


def tokenize_infix(infix):
    """
    Convert infix string into a list of tokens, handling multi-character operands.
    """
    tokens = []
    i = 0
    infix = infix.replace(' ', '')  # Remove spaces

    while i < len(infix):
        char = infix[i]

        # Multi-character alphanumeric (e.g., "abc")
        if char.isalnum() or char == '_':
            operand = char
            i += 1
            while i < len(infix) and (infix[i].isalnum() or infix[i] == '_'):
                operand += infix[i]
                i += 1
            tokens.append(operand)
            continue

        # Operators or parentheses
        elif char in '+-*/^()':
            tokens.append(char)

        else:
            raise ValueError(f"Invalid character: {char}")

        i += 1

    return tokens
