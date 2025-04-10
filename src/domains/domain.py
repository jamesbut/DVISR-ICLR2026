# A domain creates a set of data points which subsequently undergo regression.

from abc import ABC, abstractmethod
import numpy as np
from util.permutations import compute_permutations


class Domain(ABC):

    def __init__(self, config):

        self._config = config

        self._x = self.create_x()

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
    def create_x(self, num_vals=None):
        pass

    # Creates evenly spaced x values between a min and a max with a step size
    def evenly_spaced_x(self, num_vals=None):

        # If x_mins and x_maxs are defined then the independent variables
        # are created by enumerating across these ranges with x_step_sizes.
        x_mins = self._config['x_mins']
        x_maxs = self._config['x_maxs']

        # num_vals can be provided for which the step sizes must be
        # dynamically calculated
        if num_vals:

            if len(x_mins) > 1:
                raise NotImplementedError(
                    'Cannot calculate number of steps for evenly_spaced_x '
                    'for dimensions greater than 1')

            step_sizes = [(x_maxs[0] - x_mins[0]) / (num_vals - 1)]

        else:
            step_sizes = self._config['x_step_sizes']

        # Compute all permutations
        return compute_permutations(x_mins, x_maxs, step_sizes)

    # Override to return true expression
    def true_expr(self):
        return None
