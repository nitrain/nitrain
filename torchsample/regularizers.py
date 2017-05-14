
import torch as th
from fnmatch import fnmatch


class RegularizerContainer(object):

    def __init__(self, regularizers):
        self.regularizers = regularizers
        self._hooks = []
        self._model = None

    def set_model(self, model):
        self._model = model

    def register(self, model):
        for regularizer in self.regularizers:
            self.register_regularizer(model, regularizer)

    def unregister(self):
        for hook in self._hooks:
            hook.remove()

    def register_regularizer(self, model, regularizer):
        for module_name, module in self._model.named_modules():
            if fnmatch(module_name, regularizer.module_filter) and hasattr(module, 'weight'):
                hook = module.register_forward_hook(regularizer)
                self._hooks.append(hook)

    def add_regularizer(self, model, regularizer):
        self.regularizers.append(regularizer)
        self.register_regularizer(model, regularizer)

    def reset(self):
        for r in self.regularizers:
            r.reset()

    def get_loss(self):
        return sum([r.value for r in self.regularizers])

    def __len__(self):
        return len(self.regularizers)


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


# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------


class _RegularizerModule(object):

    def __init__(self, regularizers):
        self.regularizers = regularizers
        self.loss = 0.

    def _apply(self, module, regularizer):
        for name, module in module.named_children():
            if fnmatch(name, regularizer.module_filter) and hasattr(module, 'weight'):
                self.loss += regularizer(module)
                self._apply(module, regularizer)

    def __call__(self, model):
        self.loss = 0.
        for regularizer in self.regularizers:
            self._apply(model, regularizer)
        return self.loss

    def __len__(self):
        return len(self.regularizers)

