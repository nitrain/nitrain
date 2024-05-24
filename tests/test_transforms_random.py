import os
import unittest
from main import run_tests
import math
from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants
import math
from nitrain import transforms as tx

class TestClass_RandomSpatialTransforms(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_Shear(self):
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Shear((0,10))
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Shear((0,10,0))
        img2 = mytx(img)
        
        # with refernece
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Shear((0,10), img)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Shear((0,10,0), img)
        img2 = mytx(img)

    def test_RandomRotate(self):
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomRotate(-30, 30)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.RandomRotate((0,-10,0), (0,10,0))
        img2 = mytx(img)
        
        # with reference
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomRotate(-10, 10, img)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.RandomRotate((0,10,0), (0,10,0), img)
        img2 = mytx(img)
        
    def test_Zoom(self):
        img2d = ants.image_read(ants.get_data('r16'))
        img3d = ants.image_read(ants.get_data('mni'))
        
        my_tx = tx.Zoom((0.9, 0.9))
        img2d_tx = my_tx(img2d)
        
        my_tx = tx.Zoom((0.9, 0.9, 0.9))
        img3d_tx = my_tx(img3d)
        
        my_tx = tx.Zoom((1.1, 1.1))
        img2d_tx = my_tx(img2d)
        
        my_tx = tx.Zoom((1.1, 1.1, 1.1))
        img3d_tx = my_tx(img3d)
        
        with self.assertRaises(Exception):
            my_tx = tx.Zoom(1)
        
    def test_Flip(self):
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Flip()
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Flip()
        img2 = mytx(img)

    def test_Translate(self):
        img2d = ants.image_read(ants.get_data('r16'))
        img3d = ants.image_read(ants.get_data('mni'))
        
        my_tx = tx.Translate((10, 10))
        img2d_tx = my_tx(img2d)
        
        my_tx = tx.Translate((10, 10, 10))
        img3d_tx = my_tx(img3d)

        my_tx = tx.Translate((10, 0))
        img2d_tx = my_tx(img2d)
        
        my_tx = tx.Translate((10, 0, 0))   
        img3d_tx = my_tx(img3d)

        with self.assertRaises(Exception):
            my_tx = tx.Translate(1)
            

if __name__ == '__main__':
    run_tests()
