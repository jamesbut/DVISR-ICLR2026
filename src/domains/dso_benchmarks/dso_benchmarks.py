from domains.domain import Domain
import pandas as pd
import ast
import numpy as np


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

        self._expr = this_benchmark['expression']

    def evaluate(self, x):

        print(x)

        exit()

    def create_x(self, config):

        train_spec_dict = ast.literal_eval(config['train_spec'])

        if 'all' not in train_spec_dict:
            raise NotImplementedError(
                'train_spec contains values other than all'
            )

        if 'E' in train_spec_dict['all']:
            raise NotImplementedError(
                'Implement create_x for DSOBenchmarks for evenly spaced x'
            )

        if 'U' in train_spec_dict['all']:
            return np.random.uniform(
                low=train_spec_dict['all']['U'][0],
                high=train_spec_dict['all']['U'][1],
                size=train_spec_dict['all']['U'][2]
            ).reshape(-1, 1)
