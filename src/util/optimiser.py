from torch.optim import (
    RMSProp,
    Adam
)
import copy
from typing import Any, Dict


class Optimiser:
    """
    A factory class that creates PyTorch optimisers based on JSON
    configurations.
    """

    # Mapping of optimiser names to their classes
    _registry = {
        "RMSProp": RMSProp,
        "Adam": Adam,
    }

    def __init__(self, config: Dict[str, Any]):
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
        self._optimiser = OptimiserClass(**params)

    def step(self):
        self._optimiser.step()

    def zero_grad(self):
        self._optimiser.zero_grad()
