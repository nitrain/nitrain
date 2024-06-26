import os
import unittest
from main import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants
import nitrain as nt
from nitrain import samplers, transforms as tx



class TestClass_BaseSampler(unittest.TestCase):
    def setUp(self):
        img = ants.image_read(ants.get_data('mni'))
        x = [img for _ in range(5)]
        y = list(range(5))
        self.dataset = nt.Dataset(x, y)

    def tearDown(self):
        pass
    
    def test_standard(self):
        x_raw, y_raw = self.dataset[:3]
        sampler = samplers.BaseSampler(batch_size=3)
        
        sampled_batch = sampler(x_raw, y_raw)
        
        x_batch, y_batch = next(iter(sampled_batch))
        self.assertTrue(len(x_batch)==3)
        
        self.assertTrue(len(y_batch)==3)
        nptest.assert_array_equal(y_batch, np.array([0,1,2]))

class TestClass_PatchSampler(unittest.TestCase):
    def setUp(self):
        img = ants.image_read(ants.get_data('r16')).resample_image((4,4))
        x = [img for _ in range(5)]
        y = list(range(5))
        self.dataset = nt.Dataset(x, y)

    def tearDown(self):
        pass
    
    def test_standard(self):
        x_raw, y_raw = self.dataset[:3]
        sampler = samplers.PatchSampler(patch_size=(24,24),
                                        stride=(24,24),
                                        batch_size=4)
        
        sampled_batch = sampler(x_raw, y_raw)
        
        x_batch, y_batch = next(iter(sampled_batch))
        self.assertTrue(len(x_batch)==4)
        self.assertTrue(x_batch[0].shape==(24,24))
        
        self.assertTrue(len(y_batch)==4)
        # no shuffle
        self.assertTrue(all(y_batch==0))

class TestClass_SlicePatchSampler(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_standard(self):
        import nitrain as nt
        from nitrain import samplers
        import ants
        img = ants.image_read(ants.get_data('mni')).resample_image((4,4,4))
        x = [img for _ in range(5)]
        y = list(range(5))
        dataset = nt.Dataset(x, y)
        x_raw, y_raw = dataset[:5]
        sampler = samplers.SlicePatchSampler(patch_size=(32,32), 
                                             stride=(32,32), 
                                             axis=2, 
                                             batch_size=4)
        
        sampled_batch = sampler(x_raw, y_raw)
        
        x_batch, y_batch = next(iter(sampled_batch))
        self.assertTrue(len(x_batch)==4)
        self.assertTrue(x_batch[0].dimension==2)
        self.assertTrue(x_batch[0].shape==(32,32))
        
        # TODO: fix
        #self.assertTrue(len(y_batch)==4)
        #self.assertTrue(all(y_batch==0))
        
class TestClass_SliceSampler(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_standard(self):
        img = ants.image_read(ants.get_data('mni'))
        x = [img for _ in range(5)]
        y = list(range(5))
        dataset = nt.Dataset(x, y)
        x_raw, y_raw = dataset[:3]
        sampler = samplers.SliceSampler(batch_size=12, axis=2)
        
        sampled_batch = sampler(x_raw, y_raw)
        
        x_batch, y_batch = next(iter(sampled_batch))
        self.assertTrue(len(x_batch)==12)
        self.assertTrue(x_batch[0].dimension==2)
        
        
        # TODO: fix
        #self.assertTrue(len(y_batch)==12)
        #self.assertEqual(sum(y_batch), 0)

class TestClass_BlockSampler(unittest.TestCase):
    def setUp(self):
        img = ants.image_read(ants.get_data('mni'))
        x = [img for _ in range(5)]
        y = list(range(5))
        self.dataset = nt.Dataset(x, y)

    def tearDown(self):
        pass
    
    def test_standard(self):
        x_raw, y_raw = self.dataset[:3]
        sampler = samplers.BlockSampler((30,30,30), stride=(30,30,30), batch_size=12)
        sampled_batch = sampler(x_raw, y_raw)
        
        x_batch, y_batch = next(iter(sampled_batch))
        self.assertTrue(len(x_batch)==12)
        self.assertTrue(x_batch[0].shape==(30,30,30))
        
        self.assertTrue(len(y_batch)==12)
        
        # TODO: fix
        self.assertTrue(all(y_batch==0))
        
if __name__ == '__main__':
    run_tests()
