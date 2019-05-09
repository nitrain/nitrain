# python -m pytest --tb=line -svv tests/test_metrics.py
import unittest
import torch

from torchsample.metrics import CategoricalAccuracy


class TestMetrics(unittest.TestCase):

    def test_categorical_accuracy(self):
        metric = CategoricalAccuracy()
        predicted = torch.eye(10)
        expected = torch.LongTensor(list(range(10)))
        self.assertEqual(metric(predicted, expected), 100.0)

        # Set 1st column to ones
        predicted = torch.zeros(10, 10)
        predicted[:, 0] = torch.ones(10)
        self.assertEqual(metric(predicted, expected), 55.0)


if __name__ == '__main__':
    unittest.main()
