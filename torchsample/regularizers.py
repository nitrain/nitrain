
import torch as th
from fnmatch import fnmatch

from .callbacks import Callback

class RegularizerContainer(object):

    def __init__(self, regularizers):
        self.regularizers = regularizers
        self._hooks = []

    def register_forward_hooks(self, model):
        for regularizer in self.regularizers:
            for module_name, module in model.named_modules():
                if fnmatch(module_name, regularizer.module_filter) and hasattr(module, 'weight'):
                    hook = module.register_forward_hook(regularizer)
                    self._hooks.append(hook)
        
        if len(self._hooks) == 0:
            raise Exception('Tried to register regularizers but no modules '
                'were found that matched any module_filter argument.')

    def unregister_forward_hooks(self):
        for hook in self._hooks:
            hook.remove()

    def reset(self):
        for r in self.regularizers:
            r.reset()

    def get_value(self):
        value = self.regularizers[0].value
        for r in self.regularizers[1:]:
            value += r.value
        return value

    def __len__(self):
        return len(self.regularizers)


class RegularizerCallback(Callback):

    def __init__(self, container):
        self.container = container

    def on_batch_end(self, batch, logs=None):
        self.container.reset()


class ForwardHook(object):

    def reset(self):
        raise NotImplementedError('subclass must implement this method')

    def __call__(self, module, input=None, output=None):
        raise NotImplementedError('subclass must implement this method')


class L1Regularizer(ForwardHook):

    def __init__(self, scale=1e-3, module_filter='*'):
        self.scale = float(scale)
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        value = th.sum(th.abs(module.weight)) * self.scale
        self.value += value


class L2Regularizer(ForwardHook):

    def __init__(self, scale=1e-3, module_filter='*'):
        self.scale = float(scale)
        self.module_filter = module_filter
        self.value = 0.

    def reset(self):
        self.value = 0.

    def __call__(self, module, input=None, output=None):
        value = th.sum(th.pow(module.weight,2)) * self.scale
        self.value += value


class L1L2Regularizer(ForwardHook):

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
        self.value = self.l1.value + self.l2.value


