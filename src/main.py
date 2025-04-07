import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from args import get_args_parser
from config import read_config
from domains.domain_factory import create_domain
from algorithms.algorithm_factory import create_algorithm


# Plot best model found and true model if available
def plot_best_and_true_model(domain, alg, config):

    import matplotlib.pyplot as plt
    import numpy as np

    x = domain.create_x(config['domain'], num_vals=1000)

    if x.shape[1] > 1:
        print('WARNING: Cannot plot true and best models when the number '
              'of independent variables is larger than 1')
        return

    sorted_x = np.sort(x, axis=0)

    best_model = alg.best_model()

    if best_model is None:
        print('Cannot plot best model results because algorithm has not yet '
              'produced a best model')
        return

    true_model_str = domain.true_expr()

    best_y = best_model.evaluate(sorted_x)
    true_y = domain.evaluate(sorted_x)

    # Print best and true model string representations
    print('Best model:', best_model.get_infix(simplify=True))
    if true_model_str:
        print('True model:', true_model_str)

        if best_model.get_infix(simplify=True) == true_model_str:
            print('True model recovered :)')
        else:
            print('Did not recover true model :(')

    plt.plot(sorted_x, best_y, label='Best model')
    plt.plot(sorted_x, true_y, label='True model')
    plt.legend()
    plt.show()


def main():

    # Parse args
    args = get_args_parser().parse_args()

    # Read config
    config, config_path = read_config(args)

    # Create domain
    domain = create_domain(config['domain'])

    # If domain is SRBench, run their code for analysis
    if hasattr(domain, 'name') and domain.name == 'SRBench':

        # Set config path as environment variable to be used later in
        # the spaghetti SRBench process...
        os.environ['config_path'] = config_path

        # Run SRBench domain
        domain.run()

    else:

        data = domain.create_data()

        # Create algoritm
        alg = create_algorithm(config['algorithm'])

        # Perform regression
        alg.train(data)

    plot_best_and_true_model(domain, alg, config)


if __name__ == '__main__':
    main()
