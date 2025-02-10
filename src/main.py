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

    # from utils.json_helper import print_json
    # print_json(config)

    # Create domain
    domain = create_domain(config['domain'])

    data = domain.create_data()

    # Create algoritm
    alg = create_algorithm(config['algorithm'])

    # Perform regression
    alg.train(data)


if __name__ == '__main__':
    main()
