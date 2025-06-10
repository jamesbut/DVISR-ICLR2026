# This domain uses a hand written expression from the config

from domains.domain import Domain
from util.function import Function
import numpy as np


class WrittenExpression(Domain):

    def __init__(self, config):

        expr_str = config['expression']
        self._expr = Function(expr_str)

        super().__init__(config)

    def evaluate(self, x):

        # Build function argument dictionary from x values
        func_kwargs = {f'x_{i}': x[:, i] for i in range(x.shape[1])}

        # Evaluate expression
        y = self._expr(**func_kwargs)

        # Check whether y is a scalar, if so, turn into an appropriately sized
        # numpy array
        if not isinstance(y, np.ndarray):
            y = np.array([y] * x.shape[0])

        return y

    def create_x(self, num_vals=None):
        return self.evenly_spaced_x(num_vals)

    def true_expr(self):
        return self._expr.get_sympy()
