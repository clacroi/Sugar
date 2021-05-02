"""
Module representing a Caffe SGD solver.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
from google.protobuf import text_format

from sugar.solvers import Solver


class SGDSolver(Solver):
    """Class to represent a Caffe trainer for training Caffe networks,
    monitoring validation metrics and saving intermediate networks during training.

    Attributes:
        lr_policy: learning rate decay policy.
        base_lr: learning rate at beginning of training.
        gamma: learning rate decay parameter.
        momentum: value of accumulated gradients moment (SGD with momentum).
        max_iter: maximum number of training iterations.
        weight_decay: L2 weights penalization terms.
        iter_size: number of forward > backward passes per iteration.
        stepsize: for lr_policy = 'multistep', number of iterations before decreasing learning rate.
        stepvalues: for lr_policy = 'multistep', list of iteration numbers when learning rate
            should be decreased.
        cold_start: whether to start training with a reduced learning rate.
        cold_start_lr: learning rate for cold start.
        cold_start_duration: cold start duration in #iterations.
    """
    def __init__(self, lr_policy: str, base_lr: float,
                 gamma: float, momentum: float, max_iter: int,
                 weight_decay: float,
                 from_prototxt: Optional[Path] = None,
                 iter_size: Optional[int] = 1,
                 stepsize: Optional[int] = 0,
                 stepvalues: Optional[List[int]] = None,
                 cold_start: Optional[bool] = False,
                 cold_start_lr: Optional[float] = 0.01,
                 cold_start_duration : Optional[int] = 1000) -> None:
        super().__init__('SGD')

        # verify lr_policy
        if lr_policy == 'multistep':
            if stepsize != 0:
                if stepvalues:
                    raise ValueError('stepsize {} and stepvalues {} both valid.'
                                     .format(stepsize, stepvalues))
            else:
                if not stepvalues:
                    raise ValueError('stepsize {} and stepvalues {} both invalid.'
                                     .format(stepsize, stepvalues))
        else:
            raise NotImplementedError('Learning rate policy {} is not supported.'.format(lr_policy))
        self.lr_policy: str = lr_policy

        if base_lr <= 0.0:
            raise ValueError('base_lr ({}) should be > 0.0'.format(base_lr))
        self.base_lr: float = base_lr

        # verify gamma
        if gamma <= 0 or gamma > 1:
            raise ValueError('gamma ({}) should be in range ]0,1]'.format(gamma))
        self.gamma: float = gamma

        # verify momentum
        if momentum < 0 or momentum > 1:
            raise ValueError('momentum ({}) should be in range [0,1]'.format(momentum))
        self.momentum: float = momentum

        # verify max_iter
        if max_iter < 1:
            raise ValueError('max_iter ({}) should be >= 1'.format(max_iter))
        self.max_iter: int = max_iter

        # verify weight_decay
        if weight_decay < 0:
            raise ValueError('weight_decay ({}) should be >= 0.0'.format(weight_decay))
        self.weight_decay: float = weight_decay

        # verify iter_size
        if iter_size < 1:
            raise ValueError('iter_size ({}) should be >= 1'.format(iter_size))
        self.iter_size: int = iter_size

        # stepsize
        if stepsize != 0:
            if stepsize < 1:
                raise ValueError('stepsize ({}) should be >= 1'.format(stepsize))
            if stepsize > self.max_iter:
                print('Warning : stepsize ({}) is > max_iter ({})'.format(stepsize, self.max_iter))
        self.stepsize: int = stepsize

        # verify stepvalues
        self.stepvalues: List[int] = []
        if stepvalues:
            for stepvalue in sorted(stepvalues):
                if stepvalue < 1:
                    raise ValueError('stepvalue ({}) should be >= 1'.format(stepvalue))
                if stepvalue > self.max_iter:
                    print('Warning : stepvalue ({}) is > max_iter ({})'
                          .format(stepvalue, self.max_iter))
                self.stepvalues.append(stepvalue)

        # verify cold_start values
        self.cold_start: bool = cold_start
        if self.cold_start and cold_start_lr <= 0.0:
            raise ValueError('cold_start_lr ({}) should be > 0.0'.format(cold_start_lr))
        self.cold_start_lr: float = cold_start_lr

        if self.cold_start and cold_start_duration < 1:
            raise ValueError('cold_start_duration ({}) should be >= 1'.format(cold_start_duration))
        if self.cold_start and cold_start_duration > self.max_iter:
            print('Warning : cold_start_duration ({}) is > max_iter ({})'
                  .format(cold_start_duration, self.max_iter))
        self.cold_start_duration: int = cold_start_duration

        self.from_prototxt: Optional[Path] = from_prototxt

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary version json serializable"""
        return {'solver_type': type(self).__name__,
                'lr_policy': self.lr_policy,
                'base_lr': self.base_lr,
                'gamma': self.gamma,
                'momentum': self.momentum,
                'max_iter': self.max_iter,
                'weight_decay': self.weight_decay,
                'iter_size': self.iter_size,
                'stepsize': self.stepsize,
                'stepvalues': self.stepvalues,
                'cold_start': self.cold_start,
                'cold_start_lr': self.cold_start_lr,
                'cold_start_duration': self.cold_start_duration,
                'from_prototxt': str(self.from_prototxt)}

    def compute_lr(self, it: int) -> float:
        if self.cold_start and it <= self.cold_start_duration:
            return self.cold_start_lr
        if self.lr_policy == 'multistep':
            if self.stepsize:
                return self.base_lr * self.gamma ** (it // self.stepsize)
            else:
                return self.base_lr * self.gamma ** \
                    (sum([it >= stepvalue for stepvalue in self.stepvalues]))
        else:
            raise NotImplementedError('Only "multistep" lr decay policy is currently supported.')

    @classmethod
    def from_dict(cls, dictionary: Dict[str, Any]):
        """Load parameters from dictionary."""
        return cls(**dictionary)

    @classmethod
    def from_caffe_solver_protoxt(cls, caffe_solver_prototxt_file: Path):
        """Construct solver from Caffe solver prototxt file."""
        solver_param = caffe_pb2.SolverParameter()
        with open(caffe_solver_prototxt_file, 'rt') as f:
            pb2.text_format.Merge(f.read(), solver_param)
        dictionary = {'lr_policy': solver_param.lr_policy,
                      'base_lr': solver_param.base_lr,
                      'gamma': solver_param.gamma,
                      'momentum': solver_param.momentum,
                      'max_iter': solver_param.max_iter,
                      'stepsize': solver_param.stepsize,
                      'stepvalues': solver_param.stepvalue,
                      'weight_decay': solver_param.weight_decay,
                      'iter_size': solver_param.iter_size,
                      'from_prototxt': caffe_solver_prototxt_file}
        return cls(**dictionary)
