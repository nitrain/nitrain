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
        
    def test_Pad(self):
        img = nti.ones((100,110))
        mytx = tx.Pad((10,5))
        img2 = mytx(img)
        self.assertEqual(img2.shape, (120,120))

        img = nti.ones((100,110))
        mytx = tx.Pad(((10,5),5))
        img2 = mytx(img)
        self.assertEqual(img2.shape, (115,120))

        img = nti.ones((100,110))
        mytx = tx.Pad([(10,5),(3,4)])
        img2 = mytx(img)
        self.assertEqual(img2.shape, (115,117))

    def test_PadLike(self):
        img = nti.ones((100,110))

        img2 = nti.ones((72,108))
        mytx = tx.PadLike(img)
        img3 = mytx(img2)
        self.assertEqual(img.shape, img3.shape)
        
        img2 = nti.ones((73,109))
        mytx = tx.PadLike(img)
        img3 = mytx(img2)
        self.assertEqual(img.shape, img3.shape)
        
        img = nti.ones((101,111))

        img2 = nti.ones((72,108))
        mytx = tx.PadLike(img)
        img3 = mytx(img2)
        self.assertEqual(img.shape, img3.shape)
        
        img2 = nti.ones((73,109))
        mytx = tx.PadLike(img)
        img3 = mytx(img2)
        self.assertEqual(img.shape, img3.shape)


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
        
    def test_Threshold(self):
        my_tx = tx.Threshold(200)
        img_tx = my_tx(self.img_2d)

        my_tx = tx.Threshold(200, True)
        img_tx = my_tx(self.img_2d)
        
        my_tx = tx.Threshold(2000)
        img_tx = my_tx(self.img_3d)

        my_tx = tx.Threshold(2000, True)
        img_tx = my_tx(self.img_3d)
        

class TestClass_MathTransforms(unittest.TestCase):
    
    def setUp(self):
        self.img_2d = nti.ones_like(nti.example('r16'))
        self.img_3d = nti.ones_like(nti.example('mni'))

    def tearDown(self):
        pass
    
    def test_Abs(self):
        my_tx = tx.Abs()
        img2d_tx = my_tx(self.img_2d)
        img3d_tx = my_tx(self.img_3d)

    def test_Ceil(self):
        my_tx = tx.Ceil()
        img2d_tx = my_tx(self.img_2d)
        img3d_tx = my_tx(self.img_3d)

    def test_Floor(self):
        my_tx = tx.Floor()
        img2d_tx = my_tx(self.img_2d)
        img3d_tx = my_tx(self.img_3d)

    def test_Log(self):
        my_tx = tx.Log()
        img2d_tx = my_tx(self.img_2d)
        img3d_tx = my_tx(self.img_3d)

    def test_Exp(self):
        my_tx = tx.Exp()
        img2d_tx = my_tx(self.img_2d)
        img3d_tx = my_tx(self.img_3d)

    def test_Sqrt(self):
        my_tx = tx.Sqrt()
        img2d_tx = my_tx(self.img_2d)
        img3d_tx = my_tx(self.img_3d)

    def test_Power(self):
        my_tx = tx.Power(2)
        img2d_tx = my_tx(self.img_2d)
        img3d_tx = my_tx(self.img_3d)

class TestClass_ShapeTransforms(unittest.TestCase):
    
    def setUp(self):
        self.img_2d = nti.example('r16')
        self.img_3d = nti.example('mni')

    def tearDown(self):
        pass
    
    def test_Reorient(self):
        my_tx = tx.Reorient('LPI')
        img3d_tx = my_tx(self.img_3d)
        
        my_tx = tx.Reorient('IPR')
        img3d_tx = my_tx(self.img_3d)
    
    def test_Rollaxis(self):
        my_tx = tx.Rollaxis(1)
        img3d_tx = my_tx(self.img_3d)
        
        my_tx = tx.Rollaxis(2, 1)
        img3d_tx = my_tx(self.img_3d)
        
    def test_Repeat(self):
        my_tx = tx.Repeat(5)
        img2d_tx = my_tx(self.img_2d)
        img3d_tx = my_tx(self.img_3d)


class TestClass_SpatialTransforms(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_Zoom(self):
        img2d = nti.example('r16')
        img3d = nti.example('mni')
        
        my_tx = tx.Zoom(0.9)
        
        img2d_tx = my_tx(img2d)
        img3d_tx = my_tx(img3d)

        my_tx = tx.Zoom(1.1)
        
        img2d_tx = my_tx(img2d)
        img3d_tx = my_tx(img3d)
        
    def test_Flip(self):
        img2d = nti.example('r16')
        img3d = nti.example('mni')
        
        my_tx = tx.Flip(0)
        
        img2d_tx = my_tx(img2d)
        img3d_tx = my_tx(img3d)

        my_tx = tx.Flip(1)
        
        img2d_tx = my_tx(img2d)
        img3d_tx = my_tx(img3d)

if __name__ == '__main__':
    run_tests()
