import os
import unittest
from main import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants
from nitrain import datasets, loaders, samplers, transforms as tx


class TestClass_SliceSampler(unittest.TestCase):
    def setUp(self):
        img = ants.image_read(ants.get_data('mni'))
        x = [img for _ in range(5)]
        y = list(range(5))
        self.dataset = datasets.MemoryDataset(x, y)

    def tearDown(self):
        pass
    
    def test_standard(self):
        x_raw, y_raw = self.dataset[:3]
        sampler = samplers.SliceSampler(sub_batch_size=12, axis=2)
        
        sampled_batch = sampler(x_raw, y_raw)
        
        x_batch, y_batch = next(iter(sampled_batch))
        self.assertTrue(len(x_batch)==12)
        self.assertTrue(x_batch[0].dimension==2)
        
        self.assertTrue(len(y_batch)==12)
        # no shuffle
        self.assertTrue(all(y_batch==0))
        
if __name__ == '__main__':
    run_tests()
