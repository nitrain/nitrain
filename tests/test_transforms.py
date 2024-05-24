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


class TestClass_ImageTransforms(unittest.TestCase):
    def setUp(self):
        self.img_2d = ants.image_read(ants.get_data('r16'))
        self.img_3d = ants.image_read(ants.get_data('mni'))

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
        my_tx = tx.Resample((2,2))
        img_tx = my_tx(self.img_2d)
        
        my_tx = tx.Resample((2,2,2))
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
        
    def test_Pad(self):
        img = ants.from_numpy(np.ones((100,110)))
        mytx = tx.Pad((120,120))
        img2 = mytx(img)
        self.assertEqual(img2.shape, (120,120))

        img = ants.from_numpy(np.ones((100,110)))
        mytx = tx.Pad((115,120))
        img2 = mytx(img)
        self.assertEqual(img2.shape, (115,120))

        img = ants.from_numpy(np.ones((100,110)))
        mytx = tx.Pad((115,117))
        img2 = mytx(img)
        self.assertEqual(img2.shape, (115,117))


class TestClass_IntensityTransforms(unittest.TestCase):
    def setUp(self):
        self.img_2d = ants.image_read(ants.get_data('r16'))
        self.img_3d = ants.image_read(ants.get_data('mni'))

    def tearDown(self):
        pass

    def test_bias_correction(self):
        my_tx = tx.BiasCorrection()
        img_tx = my_tx(self.img_2d)
        
    def test_image_math(self):
        my_tx = tx.ImageMath('Canny', 1, 5, 12)
        img_tx = my_tx(self.img_2d)
        img_tx2 = my_tx(self.img_3d)
        
    def test_StandardNormalize(self):
        my_tx = tx.StandardNormalize()
        img_tx = my_tx(self.img_2d)
        img_tx2 = my_tx(self.img_3d)
        
    def test_RangeNormalize(self):
        my_tx = tx.RangeNormalize(0, 1)
        img_tx = my_tx(self.img_2d)
        img_tx2 = my_tx(self.img_3d)
        
        my_tx = tx.RangeNormalize(0, 2)
        img_tx = my_tx(self.img_2d)
        img_tx2 = my_tx(self.img_3d)
    
    def test_Clip(self):
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Clip(10, 200)
        img2 = mytx(img)
        self.assertEqual(img2.min(), 10)
        self.assertEqual(img2.max(), 200)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Clip(50, 52)
        img2 = mytx(img)
        self.assertEqual(img2.min(), 50)
        self.assertEqual(img2.max(), 52)
        
    def test_QuantileClip(self):
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.QuantileClip(0.1, 0.9)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.QuantileClip(0.1, 0.9)
        img2 = mytx(img)
        
    def test_Threshold(self):
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Threshold(10)
        img2 = mytx(img)
        
        self.assertEqual(img2[img2 > 0].min(), 10)
        
        mytx = tx.Threshold(200, as_upper=True)
        img2 = mytx(img)
        self.assertEqual(img2.max(), 200)


class TestFile_Utility(unittest.TestCase):
    
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_custom_function(self):
        img = ants.image_read(ants.get_data('r16'))
        def myfunc(image, val):
            return image * val
        mytx = tx.CustomFunction(myfunc, val=2.5)
        img2 = mytx(img)
        
        self.assertEqual(img2.mean() / img.mean(), 2.5)
        
    def test_numpy_function(self):
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.NumpyFunction(np.clip, a_min=10, a_max=100)
        img2 = mytx(img)
        
        self.assertEqual(img2.min(), 10)
        self.assertEqual(img2.max(), 100)

class TestClass_MathTransforms(unittest.TestCase):
    
    def setUp(self):
        self.img_2d = ants.from_numpy(np.ones((128,128)))
        self.img_3d = ants.from_numpy(np.ones((128,128,128)))

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
        self.img_2d = ants.image_read(ants.get_data('r16'))
        self.img_3d = ants.image_read(ants.get_data('mni'))

    def tearDown(self):
        pass
    
    def test_add_channel(self):
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.AddChannel()
        img2 = mytx(img)
        self.assertEqual(img2.numpy().shape, img.shape + (1,))
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.AddChannel()
        img2 = mytx(img)
        self.assertEqual(img2.numpy().shape, img.shape + (1,))
    
    def test_Reorient(self):
        img = ants.image_read(ants.get_data('mni'))
        my_tx = tx.Reorient('LPI')
        img3d_tx = my_tx(img)
        
        self.assertEqual(img3d_tx.orientation, 'LPI')
        self.assertFalse(ants.allclose(img3d_tx, img))
        
        my_tx = tx.Reorient('IPR')
        img3d_tx = my_tx(img)
        
        self.assertEqual(img3d_tx.orientation, 'IPR')
        self.assertFalse(ants.allclose(img3d_tx, img))

class TestClass_SpatialTransforms(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_apply_ants_transform(self):
        img = ants.image_read(ants.get_data('r16'))
        ants_tx = ants.new_ants_transform(dimension=2)
        ants_tx.set_parameters(ants_tx.parameters*2)
        mytx = tx.ApplyAntsTransform(ants_tx)
        img2 = mytx(img)

    def test_AffineTransform(self):
        img = ants.image_read(ants.get_data('r16'))
        theta = math.radians(90)
        arr = np.array([[math.cos(theta),-math.sin(theta), 0],
                        [math.sin(theta),math.cos(theta), 0]])
        mytx = tx.AffineTransform(arr)
        img2 = mytx(img)
        
        # with reference
        mytx = tx.AffineTransform(arr, img)
        img2 = mytx(img)

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

    def test_Rotate(self):
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Rotate(30)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Rotate((0,10,0))
        img2 = mytx(img)
        
        # with reference
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Rotate(10, img)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Rotate((0,10,0), img)
        img2 = mytx(img)
        
    def test_Zoom(self):
        img2d = ants.image_read(ants.get_data('r16'))
        img3d = ants.image_read(ants.get_data('mni'))
        
        my_tx = tx.Zoom(0.9)
        img2d_tx = my_tx(img2d)
        
        my_tx = tx.Zoom(0.9)
        img3d_tx = my_tx(img3d)
        
        my_tx = tx.Zoom(1.1)
        img2d_tx = my_tx(img2d)
        
        my_tx = tx.Zoom(1.1)
        img3d_tx = my_tx(img3d)
        
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
            
class TestLabels(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_LabelsToChannels(self):
        img2d = ants.from_numpy(np.zeros((100,100)))
        img2d[:20,:] = 1
        img2d[20:40,:] = 2
        img2d[40:60,:] = 3
        
        img3d = ants.from_numpy(np.zeros((100,100,100)))
        img3d[:20,:,:] = 1
        img3d[20:40,:,:] = 2
        img3d[40:60,:,:] = 3
        
        my_tx = tx.LabelsToChannels()

        img2d_tx = my_tx(img2d)
        img3d_tx = my_tx(img3d)
    
class TestErrors(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_base_transform(self):
        from nitrain.transforms.base import BaseTransform
        tx = BaseTransform()
        
        with self.assertRaises(NotImplementedError):
            tx.fit()
            
        with self.assertRaises(NotImplementedError):
            tx.__call__()

        with self.assertRaises(NotImplementedError):
            tx.__repr__()
        
if __name__ == '__main__':
    run_tests()
