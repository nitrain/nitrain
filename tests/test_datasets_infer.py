import os
import unittest

from tempfile import mkdtemp
import shutil
import pandas as pd

import numpy as np
import ants
import nitrain as nt

from main import run_tests

class TestClass_MemoryInferred(unittest.TestCase):
    def setUp(self):
        pass
         
    def tearDown(self):
        pass
    
    def test_single_single(self):
        imgs = [ants.image_read(ants.get_data('r16')) for _ in range(10)]
        ds = nt.Dataset(imgs, imgs)
        x, y = ds[0]
        
        self.assertTrue(ants.is_image(x))
        self.assertTrue(ants.is_image(y))
        
        x, y = ds[:2]
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 2)
        self.assertTrue(ants.is_image(x[0]))
        self.assertTrue(ants.is_image(y[0]))
        
    def test_multi_single(self):
        imgs = [ants.from_numpy(np.zeros((10,10))) for _ in range(5)]
        imgs2 = [ants.from_numpy(np.ones((10,10))) for _ in range(5)]
        ds = nt.Dataset([imgs, imgs2], imgs)
        x, y = ds[0]
        
    def test_multi_multi(self):
        imgs = [ants.from_numpy(np.zeros((10,10))) for _ in range(5)]
        imgs2 = [ants.from_numpy(np.ones((10,10))) for _ in range(5)]
        ds = nt.Dataset([imgs, imgs2], [imgs, imgs2])
        x, y = ds[0]

    def test_nested_single(self):
        import nitrain as nt
        import ants
        import numpy as np
        imgs = [ants.from_numpy(np.zeros((10,10))) for _ in range(5)]
        imgs2 = [ants.from_numpy(np.ones((10,10))) for _ in range(5)]
        imgs3 = [ants.from_numpy(np.ones((10,10)))+1 for _ in range(5)]
        ds = nt.Dataset([[imgs, imgs2], imgs3], imgs)
        x, y = ds[0]
        
        self.assertEqual(len(x), 2)
        self.assertEqual(len(x[0]), 2)
        self.assertEqual(x[0][0].mean(), 0)
        self.assertEqual(x[0][1].mean(), 1)
        self.assertEqual(x[1].mean(), 2)
        self.assertEqual(x[1].shape, (10,10))
        self.assertEqual(y.shape, (10,10))
        
    def test_nested_multi(self):
        imgs = [ants.from_numpy(np.zeros((10,10))) for _ in range(5)]
        imgs2 = [ants.from_numpy(np.ones((10,10))) for _ in range(5)]
        imgs3 = [ants.from_numpy(np.ones((10,10)))+1 for _ in range(5)]
        ds = nt.Dataset([[imgs, imgs2], imgs3], [imgs, imgs2])
        x, y = ds[0]
        
        self.assertEqual(len(x), 2)
        self.assertEqual(len(x[0]), 2)
        self.assertEqual(x[0][0].mean(), 0)
        self.assertEqual(x[0][1].mean(), 1)
        self.assertEqual(x[1].mean(), 2)
        self.assertEqual(x[1].shape, (10,10))
        self.assertEqual(y[0].shape, (10,10))
        self.assertEqual(y[1].shape, (10,10))
        
        self.assertEqual(len(y), 2)

class TestClass_MemoryDictionaryInferred(unittest.TestCase):
    def setUp(self):
        pass
         
    def tearDown(self):
        pass
    
    def test_single_single_dict(self):
        imgs = [ants.image_read(ants.get_data('r16')) for _ in range(10)]
        ds = nt.Dataset({'x': imgs}, 
                        {'y': imgs})
        x, y = ds[0]
        
        self.assertTrue(ants.is_image(x))
        self.assertTrue(ants.is_image(y))
        
        x, y = ds[:2]
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 2)
        self.assertTrue(ants.is_image(x[0]))
        self.assertTrue(ants.is_image(y[0]))
        
    def test_multi_single_dict(self):
        imgs = [ants.from_numpy(np.zeros((10,10))) for _ in range(5)]
        imgs2 = [ants.from_numpy(np.ones((10,10))) for _ in range(5)]
        ds = nt.Dataset({'x': imgs, 'y': imgs2}, {'z': imgs})
        x, y = ds[0]
        
    def test_multi_multi_dict(self):
        imgs = [ants.from_numpy(np.zeros((10,10))) for _ in range(5)]
        imgs2 = [ants.from_numpy(np.ones((10,10))) for _ in range(5)]
        ds = nt.Dataset({'x': imgs, 'y': imgs2}, {'a': imgs, 'b': imgs2})
        x, y = ds[0]

    def test_nested_single_dict(self):
        imgs = [ants.from_numpy(np.zeros((10,10))) for _ in range(5)]
        imgs2 = [ants.from_numpy(np.ones((10,10))) for _ in range(5)]
        imgs3 = [ants.from_numpy(np.ones((10,10)))+1 for _ in range(5)]
        ds = nt.Dataset({'xy': {'x': imgs, 'y': imgs2}, 'z': imgs3}, {'a': imgs})
        x, y = ds[0]
        
        self.assertEqual(len(x), 2)
        self.assertEqual(len(x[0]), 2)
        self.assertEqual(x[0][0].mean(), 0)
        self.assertEqual(x[0][1].mean(), 1)
        self.assertEqual(x[1].mean(), 2)
        self.assertEqual(x[1].shape, (10,10))
        self.assertEqual(y.shape, (10,10))
        
    def test_nested_multi_dict(self):
        imgs = [ants.from_numpy(np.zeros((10,10))) for _ in range(5)]
        imgs2 = [ants.from_numpy(np.ones((10,10))) for _ in range(5)]
        imgs3 = [ants.from_numpy(np.ones((10,10)))+1 for _ in range(5)]
        ds = nt.Dataset({'xy': {'x': imgs, 'y': imgs2}, 'z': imgs3}, 
                        {'a': imgs, 'b': imgs2})
        x, y = ds[0]
        
        self.assertEqual(len(x), 2)
        self.assertEqual(len(x[0]), 2)
        self.assertEqual(x[0][0].mean(), 0)
        self.assertEqual(x[0][1].mean(), 1)
        self.assertEqual(x[1].mean(), 2)
        self.assertEqual(x[1].shape, (10,10))
        self.assertEqual(y[0].shape, (10,10))
        self.assertEqual(y[1].shape, (10,10))
        
        self.assertEqual(len(y), 2)
        
if __name__ == '__main__':
    run_tests()
