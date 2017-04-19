
import torch

class L1Regularizer(object):
    def __init__(self, scale=0.0):
        self.scale = scale
    def __call__(self, weight):
        return torch.sum(torch.abs(weight.data)) * self.scale

class L2Regularizer(object):
    def __init__(self, scale=0.0):
        self.scale = scale
    def __call__(self, weight):
        return torch.sum(weight.data**2) * self.scale

class L1L2Regularizer(object):
    def __init__(self, l1_scale=0.0, l2_scale=0.0):
        self.l1 = L1Regularizer(l1_scale)
        self.l2 = L2Regularizer(l2_scale)
    def __call__(self, weight):
        return self.l1(weight) + self.l2(weight)

class UnitNormClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.div_(torch.norm(w, 2, 1).expand_as(w))