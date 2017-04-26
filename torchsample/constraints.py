
from __future__ import print_function
from __future__ import absolute_import

from fnmatch import fnmatch

import torch


class ConstraintModule(object):

    def __init__(self, constraints):
        self.constraints = constraints
        self.lagrangian_constraints = [c for c in self.constraints if c.lagrangian]
        if len(self.lagrangian_constraints) > 0:
            self.has_lagrangian = True
        else:
            self.has_lagrangian = False
        self.batch_constraints = [c for c in self.constraints if c.unit == 'batch']
        self.epoch_constraints = [c for c in self.constraints if c.unit == 'epoch']
        self.loss = 0.

    def set_model(self, model):
        self.model = model

    def _apply(self, module, constraint):
        for name, module in module.named_children():
            if fnmatch(name, constraint.module_filter) and hasattr(module, 'weight'):
                constraint(module)
                self._apply(module, constraint)

    def _lagrangian_apply(self, module, constraint):
        for name, module in module.named_children():
            if fnmatch(name, constraint.module_filter) and hasattr(module, 'weight'):
                self.loss += constraint(module)
                self._lagrangian_apply(module, constraint)

    def on_batch_end(self, batch):
        for constraint in self.batch_constraints:
            if ((batch+1) % constraint.frequency == 0):
                self._apply(self.model, constraint)

    def on_epoch_end(self, epoch):
        for constraint in self.epoch_constraints:
            if ((epoch+1) % constraint.frequency == 0):
                self._apply(self.model, constraint)

    def __call__(self, model):
        self.loss = 0.
        for constraint in self.lagrangian_constraints:
            self._lagrangian_apply(model, constraint)
        return self.loss

    def __len__(self):
        return len([c for c in self.constraints if c.lagrangian])



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
                 lagrangian=False,
                 scale=0.,
                 module_filter='*'):

        self.frequency = frequency
        self.unit = unit
        self.lagrangian = lagrangian
        self.module_filter = module_filter

    def __call__(self, module):
        if self.lagrangian:
            w = module.weight.data
            norm = torch.norm(w, 2, 1)
            return self.scale * torch.sum(torch.clamp(norm-1,0,1e15))
        else:
            w = module.weight.data
            w.div_(torch.norm(w, 2, 1).expand_as(w))


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
                 axis=1, 
                 frequency=1, 
                 unit='batch',
                 lagrangian=False,
                 scale=0.,
                 module_filter='*'):
        self.value = float(value)
        self.axis = axis

        self.frequency = frequency
        self.unit = unit
        self.lagrangian = lagrangian
        self.scale = scale
        self.module_filter = module_filter

    def __call__(self, module):
        if self.lagrangian:
            w = module.weight.data
            norm = torch.norm(w,2,self.axis)
            return self.scale * torch.sum(torch.clamp(norm-self.value,0,1e-15))
        else:
            w = module.weight.data
            norm = torch.norm(w,2,self.axis).expand_as(w) / self.value
            norm = torch.clamp(norm, -1e15, 1)
            w.div_(norm)


class NonNeg(Constraint):
    """
    Constrains the weights to be non-negative.
    """
    def __init__(self, 
                 frequency=1, 
                 unit='batch',
                 lagrangian=False,
                 scale=0.,
                 module_filter='*'):
        self.frequency = frequency
        self.unit = unit
        self.module_filter = module_filter

    def __call__(self, module):
        if self.lagrangian:
            w = module.weight.data
            return -1 * self.scale * torch.sum(torch.clamp(w,-1e15,0))
        else:
            w = module.weight.data
            w.clamp_(0,1e-15)






