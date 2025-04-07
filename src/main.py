import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from args import get_args_parser
from config import read_config
from domains.domain_factory import create_domain
from algorithms.algorithm_factory import create_algorithm


# Plot best model found and true model if available
def plot_best_and_true_model(domain, alg):

    if data['x'].shape[1] > 1:
        print('WARNING: Cannot plot true and best models when the number '
              'of independent variables is larger than 1')
        return

    sorted_x = np.sort(data['x'], axis=0)
    x = np.linspace(sorted_x[0][0], sorted_x[-1][0], 100)
    x = x.reshape(-1, 1)

    best_y = self._best_model.evaluate(x)
    # true_y =

    plt.plot(x, best_y)
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

    plot_best_and_true_model(domain, alg)


if __name__ == '__main__':
    main()
