"""
Abstract class representing a generic Caffe network evaluator.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import caffe


class NetworkEvaluator(ABC):
    """Abstract class to represent a .

    Attributes:
        name: name of the evaluator.
        _net: caffe network associated with the evaluator.
    """
    def __init__(self, name: str, net: caffe.Net) -> None:
        self.name = name
        self._net: caffe.Net = net

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'network_evaluator_type': type(self).__name__,
                'name': self.name}

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate network : compute and return evaluation quantities/metrics.

        Returns:
            dictionnary mapping evaluation quantities to their values.
        """
        pass

    def update_from(self, input_net: caffe.Net) -> None:
        """Update evaluator network weights from another network weights.

        Args:
            input_net: network to use for updating evaluator network weights.
        """
        params = input_net.params.keys()
        for pr in params:
            self._net.params[pr][0] = input_net.params[pr][0]
            if len(self._net.params[pr]) > 1:
                self._net.params[pr][1] = input_net.params[pr][1]
