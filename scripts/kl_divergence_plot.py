# Script that creates a plot of KL divergence against max number of tokens

import sys
import os
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
)

from domains.domain_factory import create_domain
from algorithms.algorithm_factory import create_algorithm
from algorithms.vicatsr.q import q


def calculate_kl_divs(exp_parent_dir):

    kl_divs = {}

    # For all directories in parent dir
    for exp_dir in Path(exp_parent_dir).iterdir():

        # Check whether exp_dir is directory and ends with mt_
        if exp_dir.is_dir() and bool(re.search(r'mt_.*$', str(exp_dir))):
            print(exp_dir)

            run_dir = str(exp_dir) + '/run_0'

            # Read in config
            with open(str(exp_dir) + '/config.json', 'r') as f:
                config = json.load(f)

            # Read in results
            results_file_path = run_dir + '/results.json'
            if os.path.exists(results_file_path):
                with open(results_file_path, 'r') as file:
                    results = json.load(file)
            else:
                continue

            # Filter out high max tokens
            max_tokens = config['algorithm']['max_num_tokens']
            if max_tokens > 12:
                continue

            # Create domain
            domain = create_domain(config['domain'])

            # Make sure algorithm doesn't start calculating unnecessary
            # details
            config['algorithm']['calculate_posteriors'] = False
            config['algorithm']['enum_exps'] = False
            config['algorithm']['track_kl_divergence'] = False

            # Create algorithm, data and initialise
            alg = create_algorithm(config['algorithm'], domain)
            data = domain.create_data()
            alg._initialise(data)

            # Create network paths to reflect the directory that the data is
            # currently in
            results['q']['net_path'] = os.getcwd() + '/' + run_dir + '/net.pt'

            # Read q(z)
            q_z = q.from_json(results['q'])

            # Set algorithm q as the optimised q
            alg._q = q_z

            # Calculate KL divergence
            print('Calculating KL divs...')
            kl_div = alg.kl_divergence(alg._data, num_samples=100)

            # Store kl divergence for this experiment
            kl_divs[max_tokens] = kl_div

    # Sort based on number of tokens
    kl_divs = dict(sorted(kl_divs.items()))

    return kl_divs


def plot_kl_divs(kl_divs):

    plt.bar(kl_divs.keys(), kl_divs.values())
    plt.show()


def main():

    # Take experiment parent directory as argument
    if len(sys.argv) < 2:
        print("Please provide an experiment parent directory as an argument.")
        return
    else:
        exp_parent_dir = sys.argv[1]

    exp_parent_dir = '../' + exp_parent_dir

    # Check parent dir exists
    if not os.path.isdir(exp_parent_dir):
        print(f'{exp_parent_dir} does not exist')
        return

    # Calculate kl divs
    kl_divs = calculate_kl_divs(exp_parent_dir)

    for kl in kl_divs.items():
        print(kl)

    # Create plot
    plot_kl_divs(kl_divs)


if __name__ == "__main__":
    main()
