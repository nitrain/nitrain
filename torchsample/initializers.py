"""
Classes to initialize module weights
"""

from fnmatch import fnmatch

import torch.nn.init


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

def _validate_initializer_string(init):
    dir_f = dir(torch.nn.init)
    loss_fns = [d.lower() for d in dir_f]
    if isinstance(init, str):
        try:
            str_idx = loss_fns.index(init.lower())
        except:
            raise ValueError('Invalid loss string input - must match pytorch function.')
        return getattr(torch.nn.init, dir(torch.nn.init)[str_idx])
    elif callable(init):
        return init
    else:
        raise ValueError('Invalid loss input')

class Initializer(object):

    def __init__(self, initializer, bias=False, bias_only=False, **kwargs):
        self._initializer = _validate_initializer_string(initializer)
        self.kwargs = kwargs

    def __call__(self, module):
        if self.bias_only:
            self._initializer(module.bias.data, **self.kwargs)
        else:
            self._initializer(module.weight.data, **self.kwargs)
            if self.bias:
                self._initializer(module.bias.data, **self.kwargs)


class Normal(Initializer):

    def __init__(self, mean=0.0, std=0.02, bias=False, 
                 bias_only=False, module_filter='*'):
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

