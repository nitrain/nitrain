"""
Classes to initialize module weights
"""

from fnmatch import fnmatch


class InitializerModule(object):

    def __init__(self, initializers):
        self._initializers = initializers

    def _apply(self, module, initializer):
        for name, module in module.named_children():
            if fnmatch(name, initializer.module_filter) and hasattr(module, 'weight'):
                initializer(module)
                self._apply(module, initializer)

    def __call__(self, model):
        for initializer in self._initializers:
            self._apply(model, initializer)


class Initializer(object):

    def __call__(self, module):
        raise NotImplementedError('Initializer must implement this method')


class Normal(Initializer):

    def __init__(self, mean=0.0, std=0.02, bias=False, 
                 bias_only=True, module_filter='*'):
        self.mean = mean
        self.std = std

        self.bias = bias
        self.bias_only = bias_only
        self.module_filter = module_filter

    def __call__(self, module):
        if self.bias_only:
            module.bias.data.normal_(self.mean, self.std)
        else:
            module.weight.data.normal_(self.mean, self.std)
            if self.bias:
                module.bias.data.normal_(self.mean, self.std)

