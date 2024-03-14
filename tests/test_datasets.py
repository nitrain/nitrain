import os
import unittest
from main import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants
from nitrain import datasets

class TestClass_MemoryDataset(unittest.TestCase):
    def setUp(self):
        self.img2d = ants.image_read(ants.get_data('r16'))
        self.img3d = ants.image_read(ants.get_data('mni'))

    def tearDown(self):
        pass
    
    def test_2d(self):
        x = [self.img2d for _ in range(10)]
        y = list(range(10))
        
        dataset = datasets.MemoryDataset(x, y)
        self.assertTrue(len(dataset.x) == 10)
        
    def test_3d(self):
        x = [self.img3d for _ in range(10)]
        y = list(range(10))
        
        dataset = datasets.MemoryDataset(x, y)
        self.assertTrue(len(dataset.x) == 10)


if __name__ == '__main__':
    run_tests()
