
import torch as th
from fnmatch import fnmatch

from .callbacks import Callback

class RegularizerContainer(object):

    def __init__(self, regularizers):
        self.regularizers = regularizers
        self._forward_hooks = []

    def register_forward_hooks(self, model):
        for regularizer in self.regularizers:
            for module_name, module in model.named_modules():
                if fnmatch(module_name, regularizer.module_filter) and hasattr(module, 'weight'):
                    hook = module.register_forward_hook(regularizer)
                    self._forward_hooks.append(hook)
        
        if len(self._forward_hooks) == 0:
            raise Exception('Tried to register regularizers but no modules '
                'were found that matched any module_filter argument.')

    def unregister_forward_hooks(self):
        for hook in self._forward_hooks:
            hook.remove()

    def reset(self):
        for r in self.regularizers:
            r.reset()

    def get_value(self):
        value = sum([r.value for r in self.regularizers])
        self.current_value = value.data[0]
        return value

    def __len__(self):
        return len(self.regularizers)


class RegularizerCallback(Callback):

    def __init__(self, container):
        self.container = container

    def on_batch_end(self, batch, logs=None):
        self.container.reset()


class Regularizer(object):

    def reset(self):
        raise NotImplementedError('subclass must implement this method')

    def __call__(self, module, input=None, output=None):
        raise NotImplementedError('subclass must implement this method')


class L1Regularizer(Regularizer):

    def __init__(self, scale=1e-3, module_filter='*'):
        self.scale = float(scale)
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        value = th.sum(th.abs(module.weight)) * self.scale
        self.value += value


class L2Regularizer(Regularizer):

    def __init__(self, scale=1e-3, module_filter='*'):
        self.scale = float(scale)
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        value = th.sum(th.pow(module.weight,2)) * self.scale
        self.value += value


class L1L2Regularizer(Regularizer):

    def __init__(self, l1_scale=1e-3, l2_scale=1e-3, module_filter='*'):
        self.l1 = L1Regularizer(l1_scale)
        self.l2 = L2Regularizer(l2_scale)
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        self.l1(module, input, output)
        self.l2(module, input, output)
        self.value += (self.l1.value + self.l2.value)


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

class UnitNormRegularizer(Regularizer):
    """
    UnitNorm constraint on Weights

    Constraints the weights to have column-wise unit norm
    """
    def __init__(self,
                 scale=1e-3,
                 module_filter='*'):

        self.scale = scale
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        w = module.weight
        norm_diff = th.norm(w, 2, 1).sub(1.)
        value = self.scale * th.sum(norm_diff.gt(0).float().mul(norm_diff))
        self.value += value


class MaxNormRegularizer(Regularizer):
    """
    MaxNorm regularizer on Weights

    Constraints the weights to have column-wise unit norm
    """
    def __init__(self,
                 scale=1e-3,
                 module_filter='*'):

        self.scale = scale
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        w = module.weight
        norm_diff = th.norm(w,2,self.axis).sub(self.value)
        value = self.scale * th.sum(norm_diff.gt(0).float().mul(norm_diff))
        self.value += value


class NonNegRegularizer(Regularizer):
    """
    Non-Negativity regularizer on Weights

    Constraints the weights to have column-wise unit norm
    """
    def __init__(self,
                 scale=1e-3,
                 module_filter='*'):

        self.scale = scale
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        w = module.weight
        value = -1 * self.scale * th.sum(w.gt(0).float().mul(w))
        self.value += value

