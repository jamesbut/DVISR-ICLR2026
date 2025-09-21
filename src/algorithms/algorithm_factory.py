import sys

# Import Algorithm subclasses
try:
    from algorithms.vicatsr.vicatsr import VICatSR
except ImportError as e:
    print('Could not import the VICatSR algorithm')
    print(e)


# Create algorithm from json config
def create_algorithm(config, domain=None):

    # 'name' should be set to the subclass name which is then created here
    return getattr(sys.modules[__name__], config['name'])(config, domain)
