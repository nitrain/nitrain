
from __future__ import print_function
from __future__ import absolute_import

from fnmatch import fnmatch

import torch as th
from .callbacks import Callback


class ConstraintContainer(object):

    def __init__(self, constraints):
        self.constraints = constraints
        self.batch_constraints = [c for c in self.constraints if c.unit.upper() == 'BATCH']
        self.epoch_constraints = [c for c in self.constraints if c.unit.upper() == 'EPOCH']

    def register_constraints(self, model):
        """
        Grab pointers to the weights which will be modified by constraints so
        that we dont have to search through the entire network using `apply`
        each time
        """
        # get batch constraint pointers
        self._batch_c_ptrs = {}
        for c_idx, constraint in enumerate(self.batch_constraints):
            self._batch_c_ptrs[c_idx] = []
            for name, module in model.named_modules():
                if fnmatch(name, constraint.module_filter) and hasattr(module, 'weight'):
                    self._batch_c_ptrs[c_idx].append(module)

        # get epoch constraint pointers
        self._epoch_c_ptrs = {}
        for c_idx, constraint in enumerate(self.epoch_constraints):
            self._epoch_c_ptrs[c_idx] = []
            for name, module in model.named_modules():
                if fnmatch(name, constraint.module_filter) and hasattr(module, 'weight'):
                    self._epoch_c_ptrs[c_idx].append(module)

    def apply_batch_constraints(self, batch_idx):
        for c_idx, modules in self._batch_c_ptrs.items():
            if (batch_idx+1) % self.constraints[c_idx].frequency == 0:
                for module in modules:
                    self.constraints[c_idx](module)

    def apply_epoch_constraints(self, epoch_idx):
        for c_idx, modules in self._epoch_c_ptrs.items():
            if (epoch_idx+1) % self.constraints[c_idx].frequency == 0:
                for module in modules:
                    self.constraints[c_idx](module)


class ConstraintCallback(Callback):

    def __init__(self, container):
        self.container = container

    def on_batch_end(self, batch_idx, logs):
        self.container.apply_batch_constraints(batch_idx)

    def on_epoch_end(self, epoch_idx, logs):
        self.container.apply_epoch_constraints(epoch_idx)


class Constraint(object):

    def __call__(self):
        raise NotImplementedError('Subclass much implement this method')


class UnitNorm(Constraint):
    """
    UnitNorm constraint.

    Constraints the weights to have column-wise unit norm
    """
    def __init__(self, 
                 frequency=1, 
                 unit='batch',
                 module_filter='*'):

        self.frequency = frequency
        self.unit = unit
        self.module_filter = module_filter

    def __call__(self, module):
        w = module.weight.data
        module.weight.data = w.div(th.norm(w,2,0))


class MaxNorm(Constraint):
    """
    MaxNorm weight constraint.

    Constrains the weights incident to each hidden unit
    to have a norm less than or equal to a desired value.

    Any hidden unit vector with a norm less than the max norm
    constaint will not be altered.
    """

    def __init__(self, 
                 value, 
                 axis=0, 
                 frequency=1, 
                 unit='batch',
                 module_filter='*'):
        self.value = float(value)
        self.axis = axis

        self.frequency = frequency
        self.unit = unit
        self.module_filter = module_filter

    def __call__(self, module):
        w = module.weight.data
        module.weight.data = th.renorm(w, 2, self.axis, self.value)


class NonNeg(Constraint):
    """
    Constrains the weights to be non-negative.
    """
    def __init__(self, 
                 frequency=1, 
                 unit='batch',
                 module_filter='*'):
        self.frequency = frequency
        self.unit = unit
        self.module_filter = module_filter

    def __call__(self, module):
        w = module.weight.data
        module.weight.data = w.gt(0).float().mul(w)






