import unittest
import torch
from torch.autograd import Variable

from torchsample.metrics import CategoricalAccuracy

class TestMetrics(unittest.TestCase):

    def test_categorical_accuracy(self):
        metric = CategoricalAccuracy()
        predicted = Variable(torch.eye(10))
        expected = Variable(torch.LongTensor(list(range(10))))
        self.assertEqual(metric(predicted, expected), 100.0)
        
        # Set 1st column to ones
        predicted = Variable(torch.zeros(10, 10))
        predicted.data[:, 0] = torch.ones(10)
        self.assertEqual(metric(predicted, expected), 55.0)

if __name__ == '__main__':
    unittest.main()