import sys

# Import Domain subclasses
from domains.quadratic import Quadratic


# Create domain from json config
def create_domain(config):

    # 'name' should be set to the subclass name which is then created here
    return getattr(sys.modules[__name__], config['name'])(config)
