

import os
import ants
import nitrain as nt
from nitrain import readers, samplers, transforms as tx
import os
import unittest
from main import run_tests

from tempfile import mktemp, TemporaryDirectory
import shutil

import numpy as np
import numpy.testing as nptest

import ants
import nitrain as nt
from nitrain import samplers, transforms as tx

from nitrain.samplers.slice import create_slices

class TestClass_BaseSampler(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_patch_sampler_image_to_image(self):
        import ants
        import numpy as np
        import nitrain as nt
        from nitrain import readers, transforms as tx, samplers
        imgs = [ants.from_numpy(np.random.randn(128,128)+i) for i in range(10)]
        segs = [ants.from_numpy(np.random.randn(128,128)+i).clone('unsigned int') for i in range(10)]

        # create dataset
        dataset = nt.Dataset(readers.MemoryReader(imgs),
                             readers.MemoryReader(segs),
                             transforms={
                                ('inputs', 'outputs'): tx.RandomRotate(-90, 90, p=0.1)
                             })

        ## optional: read an example record
        x, y = dataset[0]

        # strided patches
        loader = nt.Loader(dataset,
                        images_per_batch=4,
                        sampler=samplers.PatchSampler(patch_size=(96,96),
                                                        stride=(2,2),
                                                        batch_size=4),
                        transforms={
                            ('inputs', 'outputs'): tx.RandomRotate(-90, 90, p=0.5)
                        })

        xb, yb = next(iter(loader))

        self.assertEqual(xb.shape, (4,96,96,1))
        self.assertEqual(yb.shape, (4,96,96,1))
        
        # random patches
        loader = nt.Loader(dataset,
                        images_per_batch=4,
                        sampler=samplers.RandomPatchSampler(patch_size=(96,96),
                                                            patches_per_image=4,
                                                            batch_size=4),
                        transforms={
                            ('inputs', 'outputs'): tx.RandomRotate(-90, 90, p=0.5)
                        })

        xb, yb = next(iter(loader))

        self.assertEqual(xb.shape, (4,96,96,1))
        self.assertEqual(yb.shape, (4,96,96,1))
        
if __name__ == '__main__':
    run_tests()