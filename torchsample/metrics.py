

class MetricsModule(object):


    def __init__(self, metrics, prefix=''):
        self._metrics = metrics
        self._prefix = prefix

    def __call__(self, y_pred, y_true):
        logs = {self._prefix+metric._name: metric(y_pred, y_true) for metric in self._metrics}
        return logs

    def reset(self):
        for metric in self._metrics:
            metric.reset()


class Metric(object):

    def __call__(self, y_pred, y_true):
        raise NotImplementedError('Custom Metrics must implement this function')

    def reset(self):
        raise NotImplementedError('Custom Metrics must implement this function')


class CategoricalAccuracy(Metric):

    def __init__(self, top_k=1):
        self.top_k = top_k
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

        self._name = 'acc_metric'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

    def __call__(self, y_pred, y_true):
        """
        y_pred shape = (samples, classes)
        y_true shape = (samples,)
        """
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
        self.accuracy = 0

        self._name = 'acc_metric'

    def reset(self):
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

    def __call__(self, y_pred, y_true):
        """
        y_pred shape = (samples, classes)
        y_true shape = (samples,)
        """
        y_pred_round = y_pred.round().long()
        self.correct_count += y_pred_round.eq(y_true).float().sum().data[0]
        self.total_count += len(y_pred)
        accuracy = 100. * float(self.correct_count) / float(self.total_count)

        return accuracy


