import argparse
import ast
from util.json_helper import add_to_dict


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

    parser.add_argument(
        '-ca', '--config_args',
        help='Modify config via the command line'
    )

    return parser


# Modify config according to command line arguments
# Ex. --config_args "{domain/run_time:100|logging/exp_dir_name:\"exp_2\"}"
def modify_config(config, args):

    if not hasattr(args, 'config_args'):
        return config

    if args.config_args is None:
        return config

    config_args = args.config_args

    # Remove curly braces
    config_args = config_args[1:-1]

    # Split into separate JSON items
    config_items = config_args.split('|')

    # Modify config
    for item in config_items:

        # Split string to get keys and value
        keys, val = item.split(':')
        keys = keys.split('/')
        val = ast.literal_eval(val)

        # Add key value pairs to config
        add_to_dict(keys, val, config)

    return config
