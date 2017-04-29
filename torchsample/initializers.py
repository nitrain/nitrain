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

        super(Normal, self).__init__()

    def __call__(self, module):
        if self.bias_only:
            torch.nn.init.normal(module.bias.data, mean=self.mean, std=self.std)
        else:
            torch.nn.init.normal(module.weight.data, mean=self.mean, std=self.std)
            if self.bias:
                torch.nn.init.normal(module.bias.data, mean=self.mean, std=self.std)


class Uniform(Initializer):

    def __init__(self, a=0, b=1):
        self.a = a
        self.b = b

    def __call__(self, module):
        if self.bias_only:
            torch.nn.init.uniform(module.bias.data, a=self.a, b=self.b)
        else:
            torch.nn.init.uniform(module.weight.data, a=self.a, b=self.b)
            if self.bias:
                torch.nn.init.uniform(module.bias.data, a=self.a, b=self.b)


class Constant(Initializer):

    def __init__(self, val):
        self.val = val

    def __call__(self, module):
        if self.bias_only:
            torch.nn.init.constant(module.bias.data, val=self.val)
        else:
            torch.nn.init.constant(module.weight.data, val=self.val)
            if self.bias:
                torch.nn.init.constant(module.bias.data, val=self.val)


class XavierUniform(Initializer):

    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, module):
        if self.bias_only:
            torch.nn.init.xavier_uniform(module.bias.data, gain=self.gain)
        else:
            torch.nn.init.xavier_uniform(module.weight.data, gain=self.gain)
            if self.bias:
                torch.nn.init.xavier_uniform(module.bias.data, gain=self.gain)


class XavierNormal(Initializer):

    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, module):
        if self.bias_only:
            torch.nn.init.xavier_normal(module.bias.data, gain=self.gain)
        else:
            torch.nn.init.xavier_normal(module.weight.data, gain=self.gain)
            if self.bias:
                torch.nn.init.xavier_normal(module.bias.data, gain=self.gain)


class KaimingUniform(Initializer):

    def __init__(self, a=0, mode='fan_in'):
        self.a = a
        self.mode = mode

    def __call__(self, module):
        if self.bias_only:
            torch.nn.init.kaiming_uniform(module.bias.data, a=self.a, mode=self.mode)
        else:
            torch.nn.init.kaiming_uniform(module.weight.data, a=self.a, mode=self.mode)
            if self.bias:
                torch.nn.init.kaiming_uniform(module.bias.data, a=self.a, mode=self.mode)


class KaimingNormal(Initializer):

    def __init__(self, a=0, mode='fan_in'):
        self.a = a
        self.mode = mode

    def __call__(self, module):
        if self.bias_only:
            torch.nn.init.kaiming_normal(module.bias.data, a=self.a, mode=self.mode)
        else:
            torch.nn.init.kaiming_normal(module.weight.data, a=self.a, mode=self.mode)
            if self.bias:
                torch.nn.init.kaiming_normal(module.bias.data, a=self.a, mode=self.mode)


class Orthogonal(Initializer):

    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, module):
        if self.bias_only:
            torch.nn.init.orthogonal(module.bias.data, gain=self.gain)
        else:
            torch.nn.init.orthogonal(module.weight.data, gain=self.gain)
            if self.bias:
                torch.nn.init.orthogonal(module.bias.data, gain=self.gain)


class Sparse(Initializer):

    def __init__(self, sparsity, std=0.01):
        self.sparsity = sparsity
        self.std = std

    def __call__(self, module):
        if self.bias_only:
            torch.nn.init.sparse(module.bias.data, sparsity=self.sparsity, std=self.std)
        else:
            torch.nn.init.sparse(module.weight.data, sparsity=self.sparsity, std=self.std)
            if self.bias:
                torch.nn.init.sparse(module.bias.data, sparsity=self.sparsity, std=self.std)



