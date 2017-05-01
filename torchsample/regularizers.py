
import torch as th
from fnmatch import fnmatch


class RegularizerModule(object):

    def __init__(self, regularizers):
        self.regularizers = regularizers
        self.loss = 0.

    def _apply(self, module, regularizer, model_loss):
        for name, module in module.named_children():
            if fnmatch(name, regularizer.module_filter) and hasattr(module, 'weight'):
                self.loss += regularizer(module, model_loss)
                self._apply(module, regularizer)

    def __call__(self, model, model_loss):
        self.loss = 0.
        for regularizer in self.regularizers:
            self._apply(model, regularizer, model_loss)
        return self.loss

    def __len__(self):
        return len(self.regularizers)


class L1Regularizer(object):

    def __init__(self, scale=0, relative=False, module_filter='*'):
        self.scale = float(scale)
        self.relative = relative
        self.module_filter = module_filter

    def __call__(self, module, model_loss):
        w = module.weight
        value = th.sum(th.abs(w))
        if self.relative:
            loss = (self.scale * model_loss) * value
        else:
            loss = self.scale * value
        return loss


class L2Regularizer(object):

    def __init__(self, scale=0, relative=False, module_filter='*'):
        self.scale = float(scale)
        self.relative = relative
        self.module_filter = module_filter

    def __call__(self, module, model_loss):
        w = module.weight
        value = th.sum(th.pow(w,2)) * self.scale
        if self.relative:
            loss = (self.scale * model_loss) * value
        else:
            loss = self.scale * value
        return loss


class L1L2Regularizer(object):

    def __init__(self, l1_scale=0, l2_scale=0, relative=False, module_filter='*'):
        self.l1 = L1Regularizer(l1_scale, relative=relative)
        self.l2 = L2Regularizer(l2_scale, relative=relative)
        self.module_filter = module_filter

    def __call__(self, module, **kwargs):
        return self.l1(module) + self.l2(module)

