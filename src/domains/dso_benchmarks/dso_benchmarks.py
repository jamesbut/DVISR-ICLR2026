from domains.domain import Domain
import pandas as pd
import ast
import numpy as np
import re
from util.function import surround_sub_strings_with_delimiters, Function


class DSOBenchmarks(Domain):

    def __init__(self, config):

        name = config['dataset'],
        benchmarks_path = 'src/domains/dso_benchmarks/benchmarks.csv'

        # Read all DSO benchmarks from csv
        benchmarks_df = pd.read_csv(benchmarks_path, index_col=0,
                                    encoding="ISO-8859-1")

        # Retrieve the single specified benchmark
        this_benchmark = benchmarks_df.loc[name]
        self._num_vars = this_benchmark['variables']

        train_spec = this_benchmark['train_spec']

        # TODO: test spec can sometimes be different to train spec
        if not pd.isna(this_benchmark['test_spec']):
            raise NotImplementedError('DSOBenchmarks test_spec is not None')

        config['train_spec'] = train_spec

        super().__init__(config)

        # Build expression
        self._expr_str = this_benchmark['expression']

        # Surround variables with delimiters
        self._expr_str = surround_sub_strings_with_delimiters(
            self._expr_str,
            ['x' + str(i) for i in range(1, self._num_vars + 1)]
        )

        # Convert variable names from x{i} to x_{i-1}
        def shift_x_variables(expr):
            def replacer(match):
                i = int(match.group(1))
                return f"x_{i-1}"
            return re.sub(r'x(\d+)', replacer, expr)
        self._expr_str = shift_x_variables(self._expr_str)

        # Replace unary operations with numpy equivalents
        self._expr_str = self._expr_str.replace('sin', 'np.sin')
        self._expr_str = self._expr_str.replace('cos', 'np.cos')
        self._expr_str = self._expr_str.replace('exp', 'np.exp')
        self._expr_str = self._expr_str.replace('log', 'np.log')
        self._expr_str = self._expr_str.replace('sqrt', 'np.sqrt')

        # Create function from expression string
        self._expr = Function(self._expr_str)

    def evaluate(self, x):

        # Build function argument dictionary from x values
        func_kwargs = {f'x_{i}': x[:, i] for i in range(self._num_vars)}

        # Evaluate expression
        y = self._expr(**func_kwargs)

        return y

    def create_x(self, num_vals=None):

        train_spec_dict = ast.literal_eval(self._config['train_spec'])

        if 'all' not in train_spec_dict:
            raise NotImplementedError(
                'train_spec contains values other than all'
            )

        if 'E' in train_spec_dict['all']:
            raise NotImplementedError(
                'Implement create_x for DSOBenchmarks for evenly spaced x'
            )

        if 'U' in train_spec_dict['all']:

            size = (train_spec_dict['all']['U'][2] if num_vals is None
                                                   else num_vals)
            return np.random.uniform(
                low=train_spec_dict['all']['U'][0],
                high=train_spec_dict['all']['U'][1],
                size=size
            ).reshape(-1, 1)

    def true_expr(self):
        return self._expr.get_sympy()
