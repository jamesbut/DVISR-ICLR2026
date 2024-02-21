import sys

# Import Algorithm subclasses
from algorithms.symbolic_regression import SymbolicRegression


# Create algorithm from json config
def create_algorithm(config):

    # 'name' should be set to the subclass name which is then created here
    return getattr(sys.modules[__name__], config['name'])(config)
