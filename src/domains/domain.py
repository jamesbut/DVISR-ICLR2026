# A domain creates a set of data points which subsequently undergo regression.

from abc import ABC, abstractmethod
import numpy as np
from util.permutations import compute_permutations


class Domain(ABC):

    def __init__(self, config):

        self._x = self.create_x(config)

    # Create dictionary with both independent variables, x, and depdendent
    # variable y.
    # x is a 2D numpy array and y is a 1D numpy array.
    # In x, each column represents an individual dimension, each row represents
    # an individual data point.
    def create_data(self) -> dict:

        y = self.evaluate(self._x)

        return {
            "x": self._x,
            "y": y
        }

    # Each domain takes the independent variables and returns the
    # dependent variable y as a 1D numpy array.
    # x is a 2D numpy array.
    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        pass

    # Create independent variables
    @abstractmethod
    def create_x(self, config, num_vals=None):
        pass

    # Creates evenly spaced x values between a min and a max with a step size
    def evenly_spaced_x(config, num_vals=None):

        # If x_mins and x_maxs are defined then the independent variables
        # are created by enumerating across these ranges with x_step_sizes.
        if 'x_mins' in config:

            x_mins = config['x_mins']
            x_maxs = config['x_maxs']
            step_sizes = config['x_step_sizes']

        # Compute all permutations
        return compute_permutations(x_mins, x_maxs, step_sizes)
