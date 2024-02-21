from algorithms.algorithm import Algorithm
from pysr import PySRRegressor
import os


class SymbolicRegression(Algorithm):

    def __init__(self, config):

        self._regressor = PySRRegressor(
            niterations=config['num_iterations'],
            binary_operators=config['binary_operators'],
            unary_operators=config['unary_operators'],
            elementwise_loss="loss(prediction, target) = (prediction - target)^2",
            temp_equation_file=True,
            tempdir="backups",
            delete_tempfiles=False
        )

        # Create backups directory
        if not os.path.exists("backups"):
            os.makedirs("backups")

    def train(self, data):

        self._regressor.fit(data['x'], data['y'])

        #print(self._regressor)
