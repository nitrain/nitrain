


import torch.nn.functional as F
import torch.optim as optim

from ..metrics import Metric, CategoricalAccuracy, BinaryAccuracy

def validate_metric_input(metric):
    if isinstance(metric, str):
        if metric.upper() == 'CATEGORICAL_ACCURACY':
            return CategoricalAccuracy()
        elif metric.upper() == 'BINARY_ACCURACY':
            return BinaryAccuracy()
        else:
            raise ValueError('Invalid metric string input - must match torch function exactly!')
    elif isinstance(metric, Metric):
        return metric
    else:
        raise ValueError('Invalid metric input')

def validate_loss_input(loss):
    if isinstance(loss, str):
        try:
            loss_fn = eval('F.%s' % loss)
        except:
            raise ValueError('Invalid loss string input - must match torch function exactly!')
        return loss_fn
    elif callable(loss):
        return loss
    else:
        raise ValueError('Invalid metric input')

def validate_optimizer_input(optimizer):
    if isinstance(optimizer, str):
        try:
            optimizer = eval('optim.%s' % optimizer)
        except:
            raise ValueError('Invalid optimizer string input - must match torch function exactly!')
        return optimizer
    elif hasattr(optimizer, 'step') and hasattr(optimizer, 'zero_grad'):
        return optimizer
    else:
        raise ValueError('Invalid optimizer input')
