from algorithms.algorithm import Algorithm

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),
                '../../libs/DeepSymbolicOptimisation/dso'))

import libs.DeepSymbolicOptimisation.dso.dso as DSO
from DSO import DeepSymbolicRegressor


class DeepSymbolicRegression(Algorithm):

    def __init__(self, config):
        pass

    def train(self, data):
        pass
