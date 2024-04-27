import os
import unittest
from main import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ntimage as nti
from nitrain import transforms as tx


class TestClass_ImageTransforms(unittest.TestCase):
    def setUp(self):
        self.img_2d = nti.example('r16')
        self.img_3d = nti.example('mni')

    def tearDown(self):
        pass
    
    def test_Astype(self):
        my_tx = tx.Astype('float32')
        img_tx = my_tx(self.img_2d)
        img_tx2 = my_tx(self.img_3d)

    def test_Smooth(self):
        my_tx = tx.Smooth(2)
        img_tx = my_tx(self.img_2d)
        img_tx2 = my_tx(self.img_3d)

    def test_Crop(self):
        my_tx = tx.Crop()
        img_tx = my_tx(self.img_2d)
        img_tx2 = my_tx(self.img_3d)

    def test_Resample(self):
        my_tx = tx.Resample((2,2), use_spacing=True)
        img_tx = my_tx(self.img_2d)
        
        my_tx = tx.Resample((2,2,2), use_spacing=True)
        img_tx2 = my_tx(self.img_3d)

    def test_Slice(self):
        my_tx = tx.Slice(10, 0)
        img_tx = my_tx(self.img_2d)
        
        my_tx = tx.Slice(10, 0)
        img_tx2 = my_tx(self.img_3d)
        
        my_tx = tx.Slice(10, 1)
        img_tx = my_tx(self.img_2d)
        
        my_tx = tx.Slice(10, 1)
        img_tx2 = my_tx(self.img_3d)
        
        my_tx = tx.Slice(10, 2)
        img_tx2 = my_tx(self.img_3d)


class TestClass_IntensityTransforms(unittest.TestCase):
    def setUp(self):
        self.img_2d = nti.example('r16')
        self.img_3d = nti.example('mni')

    def tearDown(self):
        pass

    def test_StdNormalize(self):
        my_tx = tx.StdNormalize()
        img_tx = my_tx(self.img_2d)
        img_tx2 = my_tx(self.img_3d)
        
    def test_Normalize(self):
        my_tx = tx.Normalize(0, 1)
        img_tx = my_tx(self.img_2d)
        img_tx2 = my_tx(self.img_3d)
    
    def test_Clamp(self):
        my_tx = tx.Clamp(0, 200)
        img_tx = my_tx(self.img_2d)
        img_tx2 = my_tx(self.img_3d)

if __name__ == '__main__':
    run_tests()
