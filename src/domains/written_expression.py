# This domain uses a hand written expression from the config

from domains.domain import Domain
from utils.function import Function


class WrittenExpression(Domain):

    def __init__(self, config):

        expr_str = config['expression']
        self._expr = Function(expr_str)

        super().__init__(config)

    def evaluate(self, x):

        # Build function argument dictionary from x values
        func_kwargs = {f'x{i+1}': x[:, i] for i in range(x.shape[1])}

        # Evaluate expression
        y = self._expr(**func_kwargs)

        return y

    def create_x(self, config):
        return Domain.evenly_spaced_x(config)
