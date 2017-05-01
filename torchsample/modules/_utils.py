
import datetime
import warnings

try:
    from inspect import signature
except:
    warnings.warn('inspect.signature not available... '
        'you should upgrade to Python 3.x')

import torch.nn.functional as F
import torch.optim as optim

from ..metrics import Metric, CategoricalAccuracy, BinaryAccuracy
from ..initializers import GeneralInitializer

def _validate_metric_input(metric):
    if isinstance(metric, str):
        if metric.upper() == 'CATEGORICAL_ACCURACY' or metric.upper() == 'ACCURACY':
            return CategoricalAccuracy()
        elif metric.upper() == 'BINARY_ACCURACY':
            return BinaryAccuracy()
        else:
            raise ValueError('Invalid metric string input - must match pytorch function.')
    elif isinstance(metric, Metric):
        return metric
    else:
        raise ValueError('Invalid metric input')

def _validate_loss_input(loss):
    dir_f = dir(F)
    loss_fns = [d.lower() for d in dir_f]
    if isinstance(loss, str):
        try:
            str_idx = loss_fns.index(loss.lower())
        except:
            raise ValueError('Invalid loss string input - must match pytorch function.')
        return getattr(F, dir(F)[str_idx])
    elif callable(loss):
        return loss
    else:
        raise ValueError('Invalid loss input')

def _validate_optimizer_input(optimizer):
    dir_optim = dir(optim)
    opts = [o.lower() for o in dir_optim]
    if isinstance(optimizer, str):
        try:
            str_idx = opts.index(optimizer.lower())    
        except:
            raise ValueError('Invalid optimizer string input - must match pytorch function.')
        return getattr(optim, dir_optim[str_idx])
    elif hasattr(optimizer, 'step') and hasattr(optimizer, 'zero_grad'):
        return optimizer
    else:
        raise ValueError('Invalid optimizer input')

def _validate_initializer_input(initializer):
    if isinstance(initializer, str):
        try:
            initializer = GeneralInitializer(initializer)
        except:
            raise ValueError('Invalid initializer string input - must match pytorch function.')
        return initializer
    elif callable(initializer):
        return initializer
    else:
        raise ValueError('Invalid optimizer input')

def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")

def _nb_function_args(fn):
    return len(signature(fn).parameters)