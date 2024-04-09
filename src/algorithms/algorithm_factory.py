import sys

# Import Algorithm subclasses
try:
    from algorithms.symbolic_regression import SymbolicRegression
except ImportError as e:
    print('Could not import SymbolicRegression domain')
    print(e)

try:
    from algorithms.bayesian_machine_scientist import BayesianMachineScientist
except ImportError as e:
    print('Could not import BayesianMachineScientist domain')
    print(e)

try:
    from algorithms.deep_symbolic_regression import DeepSymbolicRegression
except ImportError as e:
    print('Could not import DeepSymbolicRegression domain')
    print(e)


# Create algorithm from json config
def create_algorithm(config):

    # 'name' should be set to the subclass name which is then created here
    return getattr(sys.modules[__name__], config['name'])(config)
