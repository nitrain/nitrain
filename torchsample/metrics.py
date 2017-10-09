
from __future__ import absolute_import
from __future__ import print_function

import torch as th

from .utils import th_matrixcorr

from .callbacks import Callback

class MetricContainer(object):


    def __init__(self, metrics, prefix=''):
        self.metrics = metrics
        self.helper = None
        self.prefix = prefix

    def set_helper(self, helper):
        self.helper = helper

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def __call__(self, output_batch, target_batch):
        logs = {}
        for metric in self.metrics:
            logs[self.prefix+metric._name] = self.helper.calculate_loss(output_batch,
                                                                        target_batch,
                                                                        metric) 
        return logs

class Metric(object):

    def __call__(self, y_pred, y_true):
        raise NotImplementedError('Custom Metrics must implement this function')

    def reset(self):
        raise NotImplementedError('Custom Metrics must implement this function')


class MetricCallback(Callback):

    def __init__(self, container):
        self.container = container
    def on_epoch_begin(self, epoch_idx, logs):
        self.container.reset()

class CategoricalAccuracy(Metric):

    def __init__(self, top_k=1):
        self.top_k = top_k
        self.correct_count = 0
        self.total_count = 0

        self._name = 'acc_metric'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, y_pred, y_true):
        top_k = y_pred.topk(self.top_k,1)[1]
        true_k = y_true.view(len(y_true),1).expand_as(top_k)
        self.correct_count += top_k.eq(true_k).float().sum().data[0]
        self.total_count += len(y_pred)
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        return accuracy


class BinaryAccuracy(Metric):

    def __init__(self):
        self.correct_count = 0
        self.total_count = 0

        self._name = 'acc_metric'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, y_pred, y_true):
        y_pred_round = y_pred.round().long()
        self.correct_count += y_pred_round.eq(y_true).float().sum().data[0]
        self.total_count += len(y_pred)
        accuracy = 100. * float(self.correct_count) / float(self.total_count)
        return accuracy


class ProjectionCorrelation(Metric):

    def __init__(self):
        self.corr_sum = 0.
        self.total_count = 0.

        self._name = 'corr_metric'

    def reset(self):
        self.corr_sum = 0.
        self.total_count = 0.

    def __call__(self, y_pred, y_true=None):
        """
        y_pred should be two projections
        """
        covar_mat = th.abs(th_matrixcorr(y_pred[0].data, y_pred[1].data))
        self.corr_sum += th.trace(covar_mat)
        self.total_count += covar_mat.size(0)
        return self.corr_sum / self.total_count


class ProjectionAntiCorrelation(Metric):

    def __init__(self):
        self.anticorr_sum = 0.
        self.total_count = 0.

        self._name = 'anticorr_metric'

    def reset(self):
        self.anticorr_sum = 0.
        self.total_count = 0.

    def __call__(self, y_pred, y_true=None):
        """
        y_pred should be two projections
        """
        covar_mat = th.abs(th_matrixcorr(y_pred[0].data, y_pred[1].data))
        upper_sum = th.sum(th.triu(covar_mat,1))
        lower_sum = th.sum(th.tril(covar_mat,-1))
        self.anticorr_sum += upper_sum
        self.anticorr_sum += lower_sum
        self.total_count += covar_mat.size(0)*(covar_mat.size(1) - 1)
        return self.anticorr_sum / self.total_count



