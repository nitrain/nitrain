
from __future__ import print_function
from __future__ import absolute_import

from fnmatch import fnmatch

import torch as th


class ConstraintContainer(object):

    def __init__(self, constraints):
        self.constraints = constraints
        #self.batch_constraints = [c for c in self.constraints if c.unit == 'batch']
        #self.epoch_constraints = [c for c in self.constraints if c.unit == 'epoch']

    def register_constraints(self, model):
        """
        Grab pointers to the weights which will be modified by constraints so
        that we dont have to search through the entire network using `apply`
        each time
        """
        self.c_ptrs = {}
        for c_idx, constraint in enumerate(self.constraints):
            self.c_ptrs[c_idx] = []
            for name, module in model.named_modules():
                if fnmatch(name, constraint.module_filter) and hasattr(module, 'weight'):
                    self.c_ptrs[c_idx].append(module)

    def apply_constraints(self):
        for c_idx, modules in self.c_ptrs.items():
            for module in modules:
                # apply constraint
                self.constraints[c_idx](module)


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
        w.div_(th.norm(w,2,1).expand_as(w))


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
        module.weight.data = th.renorm(module.weight.data, 2, self.axis, self.value)


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






