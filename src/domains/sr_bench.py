import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__),
#                 '../../libs/SRBench'))
import subprocess


class SRBench():

    def __init__(self, config):
        self.name = 'SRBench'

        # One can provide a single dataset, or if none is provided, all
        # pmlb datasets are used
        self._datasets_path = "libs/pmlb/datasets"
        if 'dataset' in config:
            self._datasets_path += '/' + config['dataset']

    def run(self):
        subprocess.run([
            "python",
            "libs/SRBench/experiment/analyze.py",
            self._datasets_path,
            "--local"
        ])
