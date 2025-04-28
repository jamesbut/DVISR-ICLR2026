import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from args import get_args_parser, modify_config
from config import read_config
from domains.domain_factory import create_domain
from algorithms.algorithm_factory import create_algorithm
from writer import Writer
from analyser import analyse_results
import time


def run_exps(args):

    # Read config
    config, config_path = read_config(args)

    # Modify config if specified via the command line
    config = modify_config(config, args)

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

        # Set file path for the initial q(z)
        init_q_file_path = writer.exp_dir_path() + '/init_net.pt'

        # Train model
        start = time.time()

        alg.train(data, init_q_file_path)

        end = time.time()
        print(f'Train time: {(end - start)/3600:.5f} hours')

    # Save results to file
    writer.write_results(alg.results())


def main():

    # Parse args
    args = get_args_parser().parse_args()

    # Analyse results
    if args.analyse:
        analyse_results(args.analyse)

    # Run experiments
    else:
        run_exps(args)


if __name__ == '__main__':
    main()
