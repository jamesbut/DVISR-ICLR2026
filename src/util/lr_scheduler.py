from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ExponentialLR,
    ReduceLROnPlateau
)
from typing import Any, Dict, Optional
import copy


class LRScheduler:
    """
    A factory class that creates PyTorch learning rate schedulers based on
    JSON configurations.
    Provides a unified interface for different scheduler types with their
    specific step requirements.
    """

    # Mapping of scheduler names to their classes
    _registry = {
        "ExponentialLR": ExponentialLR,
        "ReduceLROnPlateau": ReduceLROnPlateau,
    }

    def __init__(self, optimiser: Optimizer, config: Dict[str, Any]):
        """
        Initialize the scheduler based on the JSON config.
        """

        self._config = config
        scheduler_type = config.get('type')
        self._verbosity = config.get('verbosity', False)

        # Check scheduler is supported
        if scheduler_type not in LRScheduler._registry:
            raise ValueError(f'Unsupported scheduler type: {scheduler_type}')

        # Separate out parameters for scheduler constructor
        params = copy.deepcopy(self._config)
        del params['type']
        del params['verbosity']

        # Create scheduler
        SchedulerClass = LRScheduler._registry[scheduler_type]
        self._scheduler = SchedulerClass(optimiser.get_optimiser(), **params)

        self._optimiser = optimiser

    def step(self, metric: Optional[float] = None):
        """
        Call step function for scheduler.
        Handles special cases like ReduceLROnPlateau which needs a metric.
        """

        if isinstance(self._scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError(
                    'ReduceLROnPlateau requires a "metric" argument in step()'
                )
            self._scheduler.step(metric)
        else:
            self._scheduler.step()

        if self._verbosity:
            current_lr = self._optimiser.get_optimiser().param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6e}")
