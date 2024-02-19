import argparse


def get_args_parser():

    parser = argparse.ArgumentParser(
                prog='BayesianSymbolicRegression',
                description='Provides a testbed for Symbolic Regression experiments')
    parser.add_argument('-c', '--config', required=True)

    return parser
