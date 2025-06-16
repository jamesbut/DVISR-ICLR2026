# Rename experiment directories based on the dataset that they use.
# For example, exp_2 -> nguyen_3

import sys
import os
from pathlib import Path
import re
import json


def main():

    # Take experiment parent directory as argument
    if len(sys.argv) < 2:
        print("Please provide an experiment parent directory as an argument.")
        return
    else:
        exp_parent_dir = sys.argv[1]

    if '--dataset' not in sys.argv and '--max_num_tokens' not in sys.argv:
        print("Please provide either --dataset or --max_num_tokens as flag")
        return

    exp_parent_dir = '../' + exp_parent_dir

    # Check parent dir exists
    if not os.path.isdir(exp_parent_dir):
        print(f'{exp_parent_dir} does not exist')
        return

    # For all directories in parent dir
    for exp_dir in Path(exp_parent_dir).iterdir():

        # Check whether exp_dir is directory and ends with exp_
        if exp_dir.is_dir() and bool(re.search(r'exp_.*$', str(exp_dir))):

            # Open config.json
            with open(str(exp_dir) + '/config.json', 'r') as f:
                config = json.load(f)

            if '--dataset' in sys.argv:

                # Read dataset name
                dataset = config['domain']['dataset'].lower()

                # New exp dir name is a variation of the dataset name
                new_exp_name = dataset.lower().replace('-', '_')

            else:

                # Read max number of tokens
                max_num_tokens = config['algorithm']['max_num_tokens']

                # New exp dir name refers to the max number of tokens
                new_exp_name = 'mt_' + str(max_num_tokens)

            # Create new exp dir full path
            new_exp_path = ('/'.join(str(exp_dir).split('/')[:-1])
                            + '/' + new_exp_name)

            # Rename directory
            exp_dir.rename(Path(new_exp_path))


if __name__ == "__main__":
    main()
