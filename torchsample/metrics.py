

class MetricsModule(object):


    def __init__(self, metrics):
        self._metrics = metrics

    def __call__(self, y_pred, y_true):
        logs = {}
        for metric in self._metrics:
            logs.update(metric(y_pred, y_true))
        return logs

    def reset(self):
        for metric in self._metrics:
            metric.reset()

    def get_logs(self):
        logs = {}
        for metric in self._metrics:
            mlogs = metric.get_logs()
            logs.update(mlogs)
        return logs


class Metric(object):

    def __call__(self, y_pred, y_true):
        raise NotImplementedError('Custom Metrics must implement this function')

    def reset(self):
        raise NotImplementedError('Custom Metrics must implement this function')


class AccuracyMetric(Metric):

    def __init__(self, top_k=1):
        self.top_k = top_k
        self.correct_count = 0
        self.total_count = 0
        self.accuracy = 0

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
        #print(self.correct_count)
        self.total_count += len(y_pred)
        self.accuracy = 100. * float(self.correct_count) / float(self.total_count)

        return self.get_logs()

    def get_logs(self, prefix=''):
        return {prefix + 'acc_metric':self.accuracy}




