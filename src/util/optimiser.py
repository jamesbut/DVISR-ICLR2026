from torch.optim import (
    RMSprop,
    Adam
)
import copy


class Optimiser:
    """
    A factory class that creates PyTorch optimisers based on JSON
    configurations.
    """

    # Mapping of optimiser names to their classes
    _registry = {
        "RMSprop": RMSprop,
        "Adam": Adam,
    }

    def __init__(self, net_params, config):
        """
        Initialise the optimiser based on the JSON config.
        """

        self._config = config
        opt_type = config.get('type')

        # Check optimiser is supported
        if opt_type not in Optimiser._registry:
            raise ValueError(f'Unsupported optimiser type: {opt_type}')

        # Separate out parameters for optimiser constructor
        params = copy.deepcopy(self._config)
        del params['type']

        # Create optimiser
        OptimiserClass = Optimiser._registry[opt_type]
        self._optimiser = OptimiserClass(net_params, **params)

    def step(self):
        self._optimiser.step()

    def zero_grad(self):
        self._optimiser.zero_grad()

    def get_optimiser(self):
        return self._optimiser
