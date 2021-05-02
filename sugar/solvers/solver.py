"""
Generic module representing a Caffe solver.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Solver(ABC):
    """Class to represent a Caffe trainer for training Caffe networks,
    monitoring validation metrics and saving intermediate networks during training.

    Attributes:
        solver_type: the type of the solver.
    """
    def __init__(self, solver_type: str) -> None:
        self.solver_type: str = solver_type

    @abstractmethod
    def compute_lr(self, it: int) -> float:
        """Compute learning rate corresponding to input itration number.

        Args:
            it: current iteration number.

        Returns:
            learning rate to use for input iteration.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return self.__dict__
