import os
import unittest
from main import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants
from nitrain import datasets, loaders, samplers


class TestClass_DatasetLoader(unittest.TestCase):
    def setUp(self):
        img2d = ants.image_read(ants.get_data('r16'))
        img3d = ants.image_read(ants.get_data('mni'))
        
        x = [img2d for _ in range(10)]
        y = list(range(10))
        
        dataset_2d = datasets.MemoryDataset(x, y)
        
        x = [img3d for _ in range(10)]
        y = list(range(10))
        
        dataset_3d = datasets.MemoryDataset(x, y)
        
        self.dataset_2d = dataset_2d
        self.dataset_3d = dataset_3d
        

    def tearDown(self):
        pass
    
    def test_2d(self):
        loader = loaders.DatasetLoader(self.dataset_2d, batch_size=4)
        x_batch, y_batch = next(iter(loader))
        self.assertTrue(x_batch.shape == (4, 256, 256, 1))
    
    def test_to_keras(self):
        loader = loaders.DatasetLoader(self.dataset_2d, batch_size=4)
        keras_loader = loader.to_keras()
        x_batch, y_batch = next(iter(keras_loader))
        self.assertTrue(x_batch.shape == (4, 256, 256, 1))
        
    def test_3d(self):
        loader = loaders.DatasetLoader(self.dataset_3d,
                                       batch_size=4)

        x_batch, y_batch = next(iter(loader))
        self.assertTrue(x_batch.shape == (4, 182, 218, 182, 1))
    
    def test_image_to_image(self):
        img = ants.image_read(ants.get_data('r16'))
        x = [img for _ in range(10)]
        dataset = datasets.MemoryDataset(x, x)
        loader = loaders.DatasetLoader(dataset,
                                       batch_size=4)

        x_batch, y_batch = next(iter(loader))
        self.assertTrue(x_batch.shape == (4, 256, 256, 1))
        self.assertTrue(y_batch.shape == (4, 256, 256, 1))
    
    def test_image_to_image_with_slice_sampler(self):
        img = ants.image_read(ants.get_data('mni'))
        x = [img for _ in range(10)]
        dataset = datasets.MemoryDataset(x, x)
        loader = loaders.DatasetLoader(dataset,
                                       batch_size=4,
                                       sampler=samplers.SliceSampler(sub_batch_size=12))

        x_batch, y_batch = next(iter(loader))
        self.assertTrue(x_batch.shape == (12, 218, 182, 1))
        self.assertTrue(y_batch.shape == (12, 218, 182, 1))
    
if __name__ == '__main__':
    run_tests()
