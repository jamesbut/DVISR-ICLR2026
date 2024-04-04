import sys

# Import Algorithm subclasses
from algorithms.symbolic_regression import SymbolicRegression
from algorithms.bayesian_machine_scientist import BayesianMachineScientist
from algorithms.deep_symbolic_regression import DeepSymbolicRegression


# Create algorithm from json config
def create_algorithm(config):

    # 'name' should be set to the subclass name which is then created here
    return getattr(sys.modules[__name__], config['name'])(config)
