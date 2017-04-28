

import torch.nn.functional as F
import torch.optim as optim

from ..metrics import Metric, CategoricalAccuracy, BinaryAccuracy

def validate_metric_input(metric):
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

def validate_loss_input(loss):
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

def validate_optimizer_input(optimizer):
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
