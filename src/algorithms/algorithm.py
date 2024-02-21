# An algorithm builds a model to explain a dataset.

from abc import ABC, abstractmethod


class Algorithm(ABC):

    # Perform regression on the data.
    # data is of the same format as that returned from Domain.create_data.
    @abstractmethod
    def train(self, data: dict):
        pass
