# Regressor used for SRBench

from algorithms.vicatsr.vicatsr import VICatSR
from util.json_helper import read_json
import os

config = read_json(os.getenv('config_path'))

est = VICatSR(config['algorithm'])


def model(est, X=None):
    return est.best_model().get_infix()
