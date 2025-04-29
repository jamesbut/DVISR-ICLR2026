# A class for writing experimental results to file

import os
import glob
import json
from datetime import datetime


class Writer():

    def __init__(self, config):
        self._config = config

        default_results_dir = 'results'

        if 'writer' in self._config:
            self._write_to_file = config['writer'].get('write_to_file', True)
            results_dir = config['writer'].get('results_dir',
                                               default_results_dir)

        else:
            self._write_to_file = True
            results_dir = default_results_dir

        self._results_dir_path = os.getcwd() + '/' + results_dir

        self._initialised = False

    # Checks whether appropriate directories exist, creates them if not
    # and writes config to the directory of this experiment.
    def initialise(self):

        if not self._write_to_file:
            return

        # Create results directory if it doesn't already exist
        if not os.path.exists(self._results_dir_path):
            os.makedirs(self._results_dir_path)

        # Prepare experiment directory
        self._exp_dir_path = (self._results_dir_path + '/exp_'
                              + str(self._calculate_next_exp_dir_num()))

        # Create experiment directory
        os.makedirs(self._exp_dir_path)

        # Write config to experiment directory
        with open(self._exp_dir_path + '/config.json', 'w') as file:
            json.dump(self._config, file, indent=4)

        self._initialised = True

    # Write experiment results to file
    def write_results(self, results):

        if not self._write_to_file:
            return

        # The torch network can not be serialised so this is written
        # separately here
        net_path = self._exp_dir_path + '/net.pt'
        results['q']['net'].save(net_path)
        del results['q']['net']

        results['q']['net_path'] = net_path

        # print(json.dumps(results, indent=4))

        # Write results that could be serialised to json
        with open(self._exp_dir_path + '/results.json', 'w') as file:
            json.dump(results, file, indent=4)

        print('Experiment results written to file')

    def exp_dir_path(self):
        return self._exp_dir_path

    # Write training log line to log file
    def write_log(self, log):

        if not self._write_to_file:
            return

        if not hasattr(self, '_log_file_path'):
            self._log_file_path = self._exp_dir_path + '/log.txt'
            open(self._log_file_path, 'w').close()

        # Add timestamp to log
        log = f'[{datetime.now()}] ' + log + '\n'

        # Append log line to log file
        with open(self._log_file_path, 'a') as f:
            f.write(log)

    # Calculates the number of the directory to store exp data in
    def _calculate_next_exp_dir_num(self):

        # Search in the data directory for the next exp_ directory number

        # Get current exp directory number
        curr_exp_dir_num = self._retrieve_curr_exp_dir_num()

        # Calculate subsequent exp number
        return 0 if curr_exp_dir_num is None else curr_exp_dir_num + 1

    # Retrieve current experiment directory number
    def _retrieve_curr_exp_dir_num(self):

        exp_dirs = glob.glob(self._results_dir_path + "/*")

        if len(exp_dirs) == 0:
            return None
        else:
            max_exp_num = 0

            for ed in exp_dirs:

                # Get exp numbers in directory
                split_path = ed.split("/")
                exp_string = split_path[-1]
                try:
                    exp_num = int(exp_string.split("_")[-1])
                except ValueError:
                    continue

                # Find largest exp number
                if max_exp_num is None:
                    max_exp_num = exp_num
                else:
                    if exp_num > max_exp_num:
                        max_exp_num = exp_num

            return max_exp_num
