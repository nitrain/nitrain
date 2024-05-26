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


class TestClass_RandomImageTransforms(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_random_crop(self):
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomCrop((98,98))
        img2 = mytx(img)
        self.assertEqual(img2.shape, (98,98))
        

class TestClass_RandomSpatialTransforms(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_RandomShear(self):
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomShear((0,-10), (0,10))
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.RandomShear((0,-10,0), (0,10,0))
        img2 = mytx(img)
        
        # with refernece
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomShear((0,-10), (0,10), img)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.RandomShear((0,-10,0), (0,10,0), img)
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
        
    def test_RandomZoom(self):
        img2d = ants.image_read(ants.get_data('r16'))
        img3d = ants.image_read(ants.get_data('mni'))
        
        my_tx = tx.RandomZoom(0.9, 1.1)
        img2d_tx = my_tx(img2d)
        
        my_tx = tx.RandomZoom(0.9, 1.1)
        img3d_tx = my_tx(img3d)
        
        my_tx = tx.RandomZoom(0.9, 1.1)
        img2d_tx = my_tx(img2d)
        
        my_tx = tx.RandomZoom(0.9, 1.1)
        img3d_tx = my_tx(img3d)
        
    def test_RandomFlip(self):
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomFlip()
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.RandomFlip()
        img2 = mytx(img)

    def test_RandomTranslate(self):
        img2d = ants.image_read(ants.get_data('r16'))
        img3d = ants.image_read(ants.get_data('mni'))
        
        my_tx = tx.RandomTranslate((-10,-10), (10,10))
        img2d_tx = my_tx(img2d)
        
        my_tx = tx.RandomTranslate((-10,-10,-10), (10,10,10))
        img3d_tx = my_tx(img3d)

        my_tx = tx.RandomTranslate((0, -10), (10, 0))
        img2d_tx = my_tx(img2d)
        
        my_tx = tx.RandomTranslate((0, -10, -10), (10, 0, 0))   
        img3d_tx = my_tx(img3d)
            

if __name__ == '__main__':
    run_tests()
