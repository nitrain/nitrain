import os
import unittest
from main import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ntimage as nti
import nitrain as nt
from nitrain import samplers, transforms as tx

from nitrain.samplers.slice import create_slices

class TestClass_BaseSampler(unittest.TestCase):
    def setUp(self):
        img = nti.load(nti.example_data('mni'))
        x = [img for _ in range(5)]
        y = list(range(5))
        self.dataset = nt.Dataset(x, y)

    def tearDown(self):
        pass
    
    def test_create_slices(self):
        img = nti.zeros((5,5,5))
        img2 = nti.ones((10,10,5))
        x = create_slices([img,img,img], -1)
        self.assertEqual(len(x), 15)
        self.assertEqual(x[0].shape, (5,5))
        
        x = create_slices([[img,img2],[img,img2],[img,img2]], -1)
        self.assertEqual(len(x), 2)
        self.assertEqual(len(x[0]), 15)
        self.assertEqual(len(x[1]), 15)
        self.assertEqual(x[0][0].shape, (5,5))
        self.assertEqual(x[1][0].shape, (10,10))
        
    def test_nested_slices(self):
        img = nti.zeros((5,5,5))
        img2 = nti.ones((10,10,5))
        x3 = create_slices([[[img,img2],[img2]],[[img,img2],[img2]],[[img,img2],[img2]]],-1)
        
        self.assertEqual(len(x3), 2)
        self.assertEqual(len(x3[0]), 2)
        self.assertEqual(len(x3[0][0]), 15)
        self.assertEqual(len(x3[0][1]), 15)
        self.assertEqual(len(x3[1]), 1)
        self.assertEqual(len(x3[1][0]), 15)
        
    def test_loader_slices(self):
        import ntimage as nti
        import nitrain as nt
        from nitrain.samplers import SliceSampler
        imgs = [nti.ones((8,8,5)) for _ in range(5)]
        imgs3 = [nti.ones((12,12,5))+2 for _ in range(5)]
        ds = nt.Dataset(imgs, imgs3)
        loader = nt.Loader(ds, 
                           images_per_batch=4,
                           sampler=SliceSampler(batch_size=15, axis=-1))
        xb, yb = next(iter(loader))
        
        self.assertEqual(xb.shape, (15, 8, 8, 1))
        self.assertEqual(yb.shape, (15, 12, 12, 1))
        
        imgs = [nti.ones((8,8,5))+i for i in range(5)]
        imgs3 = [nti.ones((12,12,5))+2 for _ in range(5)]
        ds = nt.Dataset(imgs, imgs3)
        loader = nt.Loader(ds, 
                           images_per_batch=1,
                           sampler=SliceSampler(batch_size=5, axis=-1))
        
        i = 0
        for xb, yb in loader:
            i += 1
            self.assertEqual(xb.mean(), i)
        self.assertEqual(i, 5)
        
        
    def test_loader_multi_slices(self):
        import ntimage as nti
        import nitrain as nt
        from nitrain.samplers import SliceSampler
        imgs = [nti.ones((8,8,5)) for _ in range(5)]
        imgs2 = [nti.ones((10,10,5))+1 for _ in range(5)]
        imgs3 = [nti.ones((12,12,5))+2 for _ in range(5)]
        ds = nt.Dataset([imgs, imgs2], imgs3)
        
        loader = nt.Loader(ds, 
                           images_per_batch=4,
                           sampler=SliceSampler(batch_size=15, axis=-1))
        xb, yb = next(iter(loader))
        
        self.assertEqual(len(xb), 2)
        self.assertEqual(xb[0].shape, (15, 8, 8, 1))
        self.assertEqual(xb[1].shape, (15, 10, 10, 1))
        self.assertEqual(yb.shape, (15, 12, 12, 1))
        
        
if __name__ == '__main__':
    run_tests()
