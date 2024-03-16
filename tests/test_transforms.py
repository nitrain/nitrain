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


class TestClass_IntensityTransforms(unittest.TestCase):
    def setUp(self):
        self.img2d = ants.image_read(ants.get_data('r16'))
        self.img3d = ants.image_read(ants.get_data('mni'))

    def tearDown(self):
        pass
    
    def test_StandardNormalize(self):
        my_tx = tx.StandardNormalize()
        img_tx = my_tx(self.img2d)

        my_tx = tx.StandardNormalize()
        img_tx = my_tx(self.img3d)

    def test_Threshold(self):
        my_tx = tx.Threshold(0.5)
        img_tx = my_tx(self.img2d)

        my_tx = tx.Threshold(0.5)
        img_tx = my_tx(self.img3d)

    def test_RangeNormalize(self):
        my_tx = tx.RangeNormalize(0, 1)
        img_tx = my_tx(self.img2d)

        my_tx = tx.RangeNormalize(0, 1)
        img_tx = my_tx(self.img3d)

    def test_Smoothing(self):
        my_tx = tx.Smoothing(2)
        img_tx = my_tx(self.img2d)

        my_tx = tx.Smoothing(2)
        img_tx = my_tx(self.img3d)

    def test_RandomSmoothing(self):
        my_tx = tx.RandomSmoothing(1, 3)
        img_tx = my_tx(self.img2d)

        my_tx = tx.RandomSmoothing(1, 3)
        img_tx = my_tx(self.img3d)

    def test_RandomNoise(self):
        my_tx = tx.RandomNoise(0, 8)
        img_tx = my_tx(self.img2d)

        my_tx = tx.RandomNoise(0, 8)
        img_tx = my_tx(self.img3d)

    def test_HistogramWarpIntensity(self):
        my_tx = tx.HistogramWarpIntensity()
        img_tx = my_tx(self.img2d)

        my_tx = tx.HistogramWarpIntensity()
        img_tx = my_tx(self.img3d)


class TestClass_AntsTransforms(unittest.TestCase):
    def setUp(self):
        self.img2d = ants.image_read(ants.get_data('r16'))
        self.template2d = ants.image_read(ants.get_data('r64'))
        self.img3d = ants.image_read(ants.get_data('mni'))
        self.template3d = ants.image_read(ants.get_data('ch2'))

    def tearDown(self):
        pass
    
    def test_ApplyAntsTransform(self):
        atx = ants.new_ants_transform(dimension=2)
        atx.set_parameters((0.9,0,
                            0,1.1,
                            10,11))
    
        my_tx = tx.ApplyAntsTransform(atx)
        img_tx = my_tx(self.img2d)
        
        atx = ants.new_ants_transform(dimension=3)
        atx.set_parameters((0.9,0,0,
                            0,1.1,0,
                            0,0,1.2,
                            10,11,3))

        my_tx = tx.ApplyAntsTransform(atx)
        img_tx = my_tx(self.img3d)
    
    def test_BrainExtraction(self):
        my_tx = tx.BrainExtraction()
        img_tx = my_tx(self.img2d)

        my_tx = tx.BrainExtraction()
        img_tx = my_tx(self.img3d)
    
    def test_DisplacementField(self):
        my_tx = tx.DisplacementField()
        img_tx = my_tx(self.img2d)

        my_tx = tx.DisplacementField()
        img_tx = my_tx(self.img3d)
    
    def test_BiasField(self):
        my_tx = tx.BiasField()
        img_tx = my_tx(self.img2d)

        my_tx = tx.BiasField()
        img_tx = my_tx(self.img3d)
    
    def test_AlignWithTemplate(self):
        
        my_tx = tx.AlignWithTemplate(self.template2d)
        img_tx = my_tx(self.img2d)

        my_tx = tx.AlignWithTemplate(self.template3d)
        img_tx = my_tx(self.img3d)


class TestClass_SpatialTransforms(unittest.TestCase):
    def setUp(self):
        self.img2d = ants.image_read(ants.get_data('r16'))
        self.img3d = ants.image_read(ants.get_data('mni'))

    def tearDown(self):
        pass
    
    def test_RandomTranslate(self):
        my_tx = tx.RandomTranslate(-20, 20)
        img_tx = my_tx(self.img2d)
        
        my_tx = tx.RandomTranslate(-20, 20)
        img_tx = my_tx(self.img3d)
    
    def test_RandomFlip(self):
        my_tx = tx.RandomFlip(p=0.5)
        img_tx = my_tx(self.img2d)
        
        my_tx = tx.RandomFlip(p=0.5)
        img_tx = my_tx(self.img3d)
        
    def test_RandomZoom(self):
        my_tx = tx.RandomZoom(0.8, 1.2)
        img_tx = my_tx(self.img2d)
        
        my_tx = tx.RandomZoom(0.8, 1.2)
        img_tx = my_tx(self.img3d)
        
if __name__ == '__main__':
    run_tests()
