import os
import unittest
from main import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ntimage as nti
from nitrain import transforms as tx


class TestClass_StructuralTransforms(unittest.TestCase):
    def setUp(self):
        self.img_2d = nti.load(nti.example_data('r16'))
        self.img_3d = nti.load(nti.example_data('mni'))

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
        my_tx = tx.Slice(0, 10)
        img_tx = my_tx(self.img_2d)
        
        my_tx = tx.Slice(0, 10)
        img_tx2 = my_tx(self.img_3d)
        
        my_tx = tx.Slice(1, 10)
        img_tx = my_tx(self.img_2d)
        
        my_tx = tx.Slice(1, 10)
        img_tx2 = my_tx(self.img_3d)
        
        my_tx = tx.Slice(2, 10)
        img_tx2 = my_tx(self.img_3d)


if __name__ == '__main__':
    run_tests()
