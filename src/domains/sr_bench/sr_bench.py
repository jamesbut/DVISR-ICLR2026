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

        # Program skips if results have already been saved to file, one can
        # turn this off here
        self._no_skips = config.get('no_skips', False)

        # The command to be ran in order to call SRBench
        self._command = [
            "python",
            "analyze.py",
            self._datasets_path,
            "-ml", "VICatSR",
            "-script", "../../../src/domains/sr_bench/evaluate_model_main",
            "--local",
            "-results", os.getcwd() + "/srbench_results",
            # "-sym_data"
        ]

        if self._no_skips:
            self._command.append('--noskips')

    def run(self):

        # Run SRBench with arguments as if calling from the command line
        subprocess.run(self._command, cwd="libs/SRBench/experiment")
