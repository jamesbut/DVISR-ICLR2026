import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from args import get_args_parser
from config import read_config
from domains.domain_factory import create_domain
from algorithms.algorithm_factory import create_algorithm


def main():

    # Parse args
    args = get_args_parser().parse_args()

    # Read config
    config = read_config(args)

    # Create domain
    domain = create_domain(config['domain'])

    # If domain is SRBench, run their code for analysis
    if hasattr(domain, 'name') and domain.name == 'SRBench':

        domain.run()

    else:

        data = domain.create_data()

        # Create algoritm
        alg = create_algorithm(config['algorithm'])

        # Perform regression
        alg.train(data)


if __name__ == '__main__':
    main()
