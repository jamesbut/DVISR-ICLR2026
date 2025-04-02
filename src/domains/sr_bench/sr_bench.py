import os
import subprocess


class SRBench():

    def __init__(self, config):
        self.name = 'SRBench'

        # One can provide a single dataset, or if none is provided, all
        # pmlb datasets are used
        self._datasets_path = os.getcwd() + "/libs/pmlb/datasets"
        if 'dataset' in config:
            self._datasets_path += '/' + config['dataset'] + '*'

    def run(self):

        # Run SRBench with arguments as if calling from the command line
        subprocess.run([
            "python",
            "analyze.py",
            self._datasets_path,
            "-ml", "VICatSR",
            "-script", "../../../src/domains/sr_bench/evaluate_model_main",
            "--local"
        ], cwd="libs/SRBench/experiment")
