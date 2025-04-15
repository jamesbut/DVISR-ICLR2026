import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from args import get_args_parser
from config import read_config
from domains.domain_factory import create_domain
from algorithms.algorithm_factory import create_algorithm
from writer import Writer


def main():

    # Parse args
    args = get_args_parser().parse_args()

    # Read config
    config, config_path = read_config(args)

    # Save config to file
    writer = Writer(config)
    writer.initialise()

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
        alg = create_algorithm(config['algorithm'], domain)

        # Perform regression
        alg.train(data)

    # Save results to file
    writer.write_results(alg.results())


if __name__ == '__main__':
    main()
