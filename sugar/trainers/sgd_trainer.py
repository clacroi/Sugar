"""
Module to train a Caffe network with SGD.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

from datum.feeders import Feeder
import numpy as np

from sugar.trainers import Trainer
from sugar.solvers import SGDSolver
from sugar.evaluators import NetworkEvaluator


class SGDTrainer(Trainer):
    """Abstract class to represent a Caffe trainer for training Caffe networks,
    monitoring validation metrics and saving intermediate networks during training.

    Attributes:
        see Parent class.
        _accumulated_gradients: mapping from network trainable layers to their
            SGD accumulated gradients.
    """
    def __init__(self, root_dir: Path,
                 train_net_prototxt: Path, trainable_layers: List[str],
                 train_net_weights: Optional[Path] = None,
                 train_input_loading_mode: Optional[str] = 'CUSTOM',
                 train_feeder: Optional[Feeder] = None,
                 evaluators: Optional[List[NetworkEvaluator]] = None,
                 solver: Optional[SGDSolver] = None,
                 caffe_solver_prototxt: Optional[Path] = None,
                 display_interval: Optional[int] = 50,
                 eval_interval: Optional[int] = 500,
                 snapshot_interval: Optional[int] = 2000,
                 caffe_mode: Optional[str] = 'GPU') -> None:
        # Load solver
        if solver.solver_type != 'SGD':
            raise ValueError('SGDTrainer should be initalized with'
                             'solver having type "SGD" vs. ({})'
                             .format(solver.solver_type))
        if not solver:
            if not caffe_solver_prototxt:
                raise ValueError('At least 1 parameter among solver and '
                                 'caffe_solver_protoxt must be passed to initialize Trainer.')
            solver = SGDSolver.from_caffe_solver_protoxt(caffe_solver_prototxt)
        super().__init__(root_dir,
                         train_net_prototxt, trainable_layers, train_net_weights,
                         train_input_loading_mode, train_feeder,
                         evaluators,
                         solver, display_interval, eval_interval, snapshot_interval,
                         caffe_mode)

        # Construct dictionary gathering weights updates at each iteration
        self._accumulated_gradients: Dict[str, np.ndarray] = {}
        for layer_name, layer in self._train_net.params.items():
            self._accumulated_gradients[layer_name] = [0.0 for _ in layer]

    def _update_network_weights(self, it: int, lr: float) -> None:
        """Update training net weights with current gradients following the
        Stochastic Gradient Descent with momentum algorithm.

        Args:
            it: current step number.
            lr: learning rate to use for current step.

        Returns:
            Output of Caffe training net forward pass.
        """
        for layer_name, layer_weights in self._train_net.params.items():
            layer_params = self._train_net_params[layer_name]
            if layer_params.type not in self.trainable_layers:
                continue
            for i, blob in enumerate(layer_weights):
                if len(layer_params.param) > i:
                    lr_mult = layer_params.param[i].lr_mult
                    decay_mult = layer_params.param[i].decay_mult
                else:
                    lr_mult = 1.0
                    decay_mult = 1.0

                # Case lr_mult = 0, weights should not be updated
                if lr_mult == 0:
                    continue

                # Compute current gradient
                # (i) Compute loss gradient
                current_gradient = blob.diff / float(self.solver.iter_size)

                # (ii) Add weights penalization term if necessary
                if decay_mult > 0:
                    current_gradient += self.solver.weight_decay * decay_mult * blob.data

                # Update accumulated gradients :
                # momentum * accumulated gradient + (1 - momentum) * current gradient
                if it == 0:
                    self._accumulated_gradients[layer_name][i] = current_gradient
                else:
                    self._accumulated_gradients[layer_name][i] =\
                        self.solver.momentum * self._accumulated_gradients[layer_name][i] +\
                        (1 - self.solver.momentum) * current_gradient

                # Update weights
                blob.data[...] = blob.data - lr * lr_mult * self._accumulated_gradients[layer_name][i]
