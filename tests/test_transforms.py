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
        self.img2d = nti.load(nti.example_data('r16'))
        self.img3d = nti.load(nti.example_data('mni'))

    def tearDown(self):
        pass
    
    def test_Resample(self):
        my_tx = tx.Resample((2,2), use_spacing=True)
        img_tx = my_tx(self.img2d)
        
        my_tx = tx.Resample((2,2,2), use_spacing=True)
        img_tx = my_tx(self.img3d)

        
if __name__ == '__main__':
    run_tests()
