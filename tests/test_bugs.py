import os
import unittest

from tempfile import mkdtemp
import shutil
import pandas as pd

import numpy as np
import ants
import nitrain as nt
from nitrain import readers, transforms as tx
        
from main import run_tests

class TestClass_Bugs(unittest.TestCase):
    def setUp(self):
        pass
         
    def tearDown(self):
        pass
    
    def test_random_transform_applies_to_both_images(self):
        import ants
        import numpy as np
        import nitrain as nt
        from nitrain import readers, transforms as tx
        from nitrain.transforms.base import BaseTransform
        
        import random
        
        imgs = [ants.from_numpy(np.zeros((100,100))) for i in range(10)]
        for i in range(10):
            imgs[i][30:70,30:70] = 1
        
        class RandomAddition(BaseTransform):
            def __init__(self):
                pass
            def __call__(self, *images):
                value = int(random.uniform(0, 1) * 100)
                new_images = [image + value for image in images]
                return new_images if len(new_images) > 1 else new_images[0]
    
        # create dataset
        dataset = nt.Dataset(inputs=readers.MemoryReader(imgs),
                             outputs=readers.MemoryReader(imgs),
                             transforms={
                                ('inputs', 'outputs'): RandomAddition()
                            })
        x, y = dataset[0]
        self.assertTrue(ants.allclose(x, y))
        
        loader = nt.Loader(dataset,
                        images_per_batch=4,
                        transforms={
                            ('inputs', 'outputs'): RandomAddition()
                        })
        xb, yb = next(iter(loader))
        self.assertTrue(np.allclose(xb, yb))
        
        
        dataset = nt.Dataset(inputs=readers.MemoryReader(imgs),
                             outputs=readers.MemoryReader(imgs),
                             transforms={
                                ('inputs', 'outputs'): tx.RandomRotate(-90, 90, p=1)
                            })
        x, y = dataset[0]
        self.assertTrue(ants.allclose(x, y))


        loader = nt.Loader(dataset,
                        images_per_batch=4,
                        transforms={
                            ('inputs', 'outputs'): tx.RandomRotate(-90, 90, p=1)
                        })
        xb, yb = next(iter(loader))
        self.assertTrue(np.allclose(xb, yb))
        
        
        loader = nt.Loader(dataset,
                        images_per_batch=4,
                        transforms={
                            ('inputs', 'outputs'): tx.RandomRotate(-90, 90, p=1),
                            'inputs': tx.RandomRotate(-90, 90, p=1),
                            'outputs': tx.RandomRotate(-90, 90, p=1)
                        })
        xb, yb = next(iter(loader))
        self.assertFalse(np.allclose(xb, yb))

    
if __name__ == '__main__':
    run_tests()
