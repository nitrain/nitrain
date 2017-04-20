
import torch
from fnmatch import fnmatch

class RegularizerList(object):

    def __init__(self, regularizers):
        self.regularizers = regularizers
        self.loss = 0.

    def set_model(self, model):
        self.model = model

    def _apply(self, module, regularizer):
        for name, module in module.named_children():
            if fnmatch(name, regularizer.module_filter) and hasattr(module, 'weight'):
                self.loss += regularizer(module)
                self._apply(module, regularizer)

    def compute_loss(self):
        self.loss = 0.
        for regularizer in self.regularizers:
            self._apply(self.model, regularizer)
        return self.loss

    def __len__(self):
        return len(self.regularizers)

class L1Regularizer(object):

    def __init__(self, scale=0.0, module_filter='*'):
        self.scale = scale
        self.module_filter = module_filter

    def __call__(self, module):
        w = module.weight.data
        return torch.sum(torch.abs(w)) * self.scale

class L2Regularizer(object):

    def __init__(self, scale=0.0, module_filter='*'):
        self.scale = scale
        self.module_filter = module_filter

    def __call__(self, module):
        w = module.weight.data
        return torch.sum(torch.pow(w,2)) * self.scale

class L1L2Regularizer(object):

    def __init__(self, l1_scale=0.0, l2_scale=0.0, module_filter='*'):
        self.l1 = L1Regularizer(l1_scale)
        self.l2 = L2Regularizer(l2_scale)
        self.module_filter = module_filter

    def __call__(self, module):
        return self.l1(module) + self.l2(module)

