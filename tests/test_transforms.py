import os
import unittest
from main import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants
from nitrain import transforms as tx


class TestClass_StructuralTransforms(unittest.TestCase):
    def setUp(self):
        self.img2d = ants.image_read(ants.get_data('r16'))
        self.img3d = ants.image_read(ants.get_data('mni'))

    def tearDown(self):
        pass
    
    def test_Resample(self):
        my_tx = tx.Resample((2,2), use_spacing=True)
        img_tx = my_tx(self.img2d)
        
        my_tx = tx.Resample((2,2,2), use_spacing=True)
        img_tx = my_tx(self.img3d)
    
    def test_ResampleToTarget(self):
        my_tx = tx.ResampleToTarget(self.img2d.resample_image((3,3)))
        img_tx = my_tx(self.img2d)
        
        my_tx = tx.ResampleToTarget(self.img3d.resample_image((3,3,3)))
        img_tx = my_tx(self.img3d)
        
    def test_Reorient(self):
        my_tx = tx.Reorient('RAS')
        with self.assertRaises(Exception):
            img_tx = my_tx(self.img2d)

        my_tx = tx.Reorient('RAS')
        img_tx = my_tx(self.img3d)
        
    def test_Slice(self):
        my_tx = tx.Slice(2, 120)
        with self.assertRaises(Exception):
            img_tx = my_tx(self.img2d)

        my_tx = tx.Slice(2, 120)
        img_tx = my_tx(self.img3d)
        
    def test_RandomSlice(self):
        my_tx = tx.RandomSlice(2)
        with self.assertRaises(Exception):
            img_tx = my_tx(self.img2d)

        my_tx = tx.RandomSlice(2)
        img_tx = my_tx(self.img3d)
        self.assertTrue(img_tx.shape == (182, 218))
        
    def test_Crop(self):
        my_tx = tx.Crop([0,0], [20, 20])
        img_tx = my_tx(self.img2d)
        self.assertTrue(img_tx.shape == (20,20))

        my_tx = tx.Crop([0,0,0], [20,20,20])
        img_tx = my_tx(self.img3d)
        self.assertTrue(img_tx.shape == (20,20,20))
        
    def test_RandomCrop(self):
        my_tx = tx.RandomCrop([30,30])
        img_tx = my_tx(self.img2d)
        self.assertTrue(img_tx.shape == (30,30))

        my_tx = tx.RandomCrop([30,30,30])
        img_tx = my_tx(self.img3d)
        self.assertTrue(img_tx.shape == (30,30,30))
        
    def test_Pad(self):
        my_tx = tx.Pad(shape=[300,300])
        img_tx = my_tx(self.img2d)
        self.assertTrue(img_tx.shape == (300,300))

        my_tx = tx.Pad(shape=[300,300,300])
        img_tx = my_tx(self.img3d)
        self.assertTrue(img_tx.shape == (300,300,300))


if __name__ == '__main__':
    run_tests()
