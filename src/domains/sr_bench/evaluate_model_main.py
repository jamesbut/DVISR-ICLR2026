import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),
                '../../../libs/SRBench/experiment/'))
sys.path.append(os.path.join(os.path.dirname(__file__),
                '../../../src/'))
from evaluate_model import set_env_vars, evaluate_model

lib_name = "algorithms.vicatsr.regressor"


# This main is copied from libs/SRBench/experiment/evaluate_model.py
# I have altered it slightly so that it can read regressor files that are
# not just in libs/SRBench/methods/

################################################################################
# main entry point
################################################################################
import argparse
import importlib

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument(
        '-ml', action='store', dest='ALG', default=None,
        type=str, help='Name of estimator (with matching file in methods/)')
    parser.add_argument('-results_path', action='store', dest='RDIR',
                        default='results_test', type=str,
                        help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
                        default=42, type=int, help='Seed / trial')
    parser.add_argument('-test', action='store_true', dest='TEST',
                        help='Used for testing a minimal version')
    parser.add_argument('-n_jobs', action='store', type=str, default='4',
                        help='number of cores available')
    parser.add_argument('-max_samples', action='store', type=int, default=0,
                        help='number of training samples')
    parser.add_argument('-target_noise', action='store', dest='Y_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-feature_noise', action='store', dest='X_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-sym_data', action='store_true',
                        help='Use symbolic dataset settings')
    parser.add_argument('-skip_tuning', action='store_true', dest='SKIP_TUNE',
                        default=False, help='Dont tune the estimator')

    args = parser.parse_args()

    set_env_vars(args.n_jobs)

    # Import algorithm
    print(f'import from {lib_name}')
    algorithm = importlib.__import__(lib_name, globals(), locals(), ['*'])

    print('algorithm:', algorithm.est)

    # Optional keyword arguments passed to evaluate
    eval_kwargs, test_params = {}, {}
    if 'eval_kwargs' in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs

    if args.max_samples != 0:
        eval_kwargs['max_train_samples'] = args.max_samples

    evaluate_model(args.INPUT_FILE,
                   args.RDIR,
                   args.RANDOM_STATE,
                   args.ALG,
                   algorithm.est,
                   algorithm.model,
                   test = args.TEST,
                   **eval_kwargs)
