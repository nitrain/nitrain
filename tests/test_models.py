import os
import unittest
from main import run_tests

from tempfile import mktemp, mkdtemp
import shutil

import json
import pandas as pd
import numpy as np
import numpy.testing as nptest

import ntimage as nt
from nitrain import models


class TestClass_Models(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_fetch_architecture(self):
        arch_fn = models.fetch_architecture('vgg', dim=2)
        model = arch_fn((48,48,1))
        
        arch_fn = models.fetch_architecture('vgg', dim=3)
        model = arch_fn((48,48,48,1))
    
    def test_list_architectures(self):
        archs = models.list_architectures()
        self.assertTrue(len(archs) > 0)

if __name__ == '__main__':
    run_tests()
