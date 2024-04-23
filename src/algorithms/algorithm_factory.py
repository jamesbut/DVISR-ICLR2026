import sys

# Import Algorithm subclasses
try:
    from algorithms.symbolic_regression import SymbolicRegression
except ImportError as e:
    print('Could not import the SymbolicRegression algorithm')
    print(e)

try:
    from algorithms.bayesian_machine_scientist import BayesianMachineScientist
except ImportError as e:
    print('Could not import the BayesianMachineScientist algorithm')
    print(e)

try:
    from algorithms.deep_symbolic_regression import DeepSymbolicRegression
except ImportError as e:
    print('Could not import the DeepSymbolicRegression algorithm')
    print(e)

try:
    from algorithms.ddsr import DDSR
except ImportError as e:
    print('Could not import the DDSR algorithm')
    print(e)

try:
    from algorithms.vicatsr import VICatSR
except ImportError as e:
    print('Could not import the VICatSR algorithm')
    print(e)


# Create algorithm from json config
def create_algorithm(config):

    # 'name' should be set to the subclass name which is then created here
    return getattr(sys.modules[__name__], config['name'])(config)
