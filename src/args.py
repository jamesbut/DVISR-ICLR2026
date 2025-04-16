import argparse


def get_args_parser():

    parser = argparse.ArgumentParser(
        prog='BayesianSymbolicRegression',
        description='Provides a testbed for Symbolic Regression experiments')

    # Run experiments with a configuration file
    parser.add_argument('-c', '--config')

    # Analyse results
    parser.add_argument(
        '-a', '--analyse',
        help='Provide experiment directory to analyse, e.g. results/exp_4'
    )

    return parser
