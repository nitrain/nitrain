import os
import unittest
from main import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ntimage as nti
import nitrain as nt
from nitrain import samplers, readers


class TestClass_DatasetLoader(unittest.TestCase):
    def setUp(self):
        img2d = nti.load(nti.example_data('r16'))
        img3d = nti.load(nti.example_data('mni'))
        
        x = [img2d for _ in range(10)]
        y = list(range(10))
        
        dataset_2d = nt.Dataset(x, y)
        
        x = [img3d for _ in range(10)]
        y = list(range(10))
        
        dataset_3d = nt.Dataset(x, y)
        
        self.dataset_2d = dataset_2d
        self.dataset_3d = dataset_3d
        
    def tearDown(self):
        pass
    
    def test_2d(self):
        loader = nt.Loader(self.dataset_2d, images_per_batch=4)
        x_batch, y_batch = next(iter(loader))
        self.assertTrue(x_batch.shape == (4, 256, 256, 1))
    
    def test_to_keras(self):
        loader = nt.Loader(self.dataset_2d, images_per_batch=4)
        keras_loader = loader.to_keras()
        x_batch, y_batch = next(iter(keras_loader))
        self.assertTrue(x_batch.shape == (4, 256, 256, 1))
        
    def test_3d(self):
        loader = nt.Loader(self.dataset_3d, images_per_batch=4)

        x_batch, y_batch = next(iter(loader))
        self.assertTrue(x_batch.shape == (4, 182, 218, 182, 1))
    
    def test_image_to_image(self):
        img = nti.load(nti.example_data('r16'))
        x = [img for _ in range(10)]
        dataset = nt.Dataset(x, x)
        loader = nt.Loader(dataset, images_per_batch=4)

        x_batch, y_batch = next(iter(loader))
        self.assertTrue(x_batch.shape == (4, 256, 256, 1))
        self.assertTrue(y_batch.shape == (4, 256, 256, 1))

    def test_multi_image_to_image(self):
        img = nti.load(nti.example_data('r16'))
        dataset = nt.Dataset([[img, img] for _ in range(10)], 
                             [img for _ in range(10)])
        loader = nt.Loader(dataset, images_per_batch=4)

        x_batch, y_batch = next(iter(loader))
        self.assertTrue(len(x_batch) == 2)
        self.assertTrue(x_batch[0].shape == (4, 256, 256, 1))
        self.assertTrue(x_batch[1].shape == (4, 256, 256, 1))
        self.assertTrue(y_batch.shape == (4, 256, 256, 1))
    
    def test_image_to_image_with_slice_sampler(self):
        img = nti.load(nti.example_data('mni'))
        x = [img for _ in range(10)]
        dataset = nt.Dataset(x, x)
        loader = nt.Loader(dataset, 
                           images_per_batch=4,
                           sampler=samplers.SliceSampler(batch_size=12))

        x_batch, y_batch = next(iter(loader))
        self.assertTrue(x_batch.shape == (12, 218, 182, 1))
        self.assertTrue(y_batch.shape == (12, 218, 182, 1))
        
    def test_multiple_image_slice(self):
        base_dir = nt.fetch_data('example-01')

        dataset = nt.Dataset(inputs={'a': readers.PatternReader('*/img3d.nii.gz'),
                                    'b': readers.PatternReader('*/img3d_100.nii.gz')},
                            outputs=readers.PatternReader('*/img3d_seg.nii.gz'),
                            base_dir=base_dir)

        loader = nt.Loader(dataset,
                           images_per_batch=1,
                           sampler=samplers.SliceSampler(batch_size=50, axis=-1))

        for i, (xbatch, ybatch) in enumerate(loader):
                self.assertEqual(xbatch[0].mean(), i+1)
                self.assertEqual(xbatch[1].mean(), i+1+100)
    
if __name__ == '__main__':
    run_tests()
