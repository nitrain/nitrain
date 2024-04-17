import os
import unittest
from main import run_tests

from tempfile import mktemp, mkdtemp, NamedTemporaryFile
import shutil

import base64
import json
import pandas as pd
import numpy as np
import numpy.testing as nptest

import ntimage as nt
import nitrain
from nitrain import readers

class TestClass_Dataset(unittest.TestCase):
    def setUp(self):
        pass
         
    def tearDown(self):
        pass
    
    def test_memory(self):
        img = nt.load(nt.example_data('r16'))
        dataset = nitrain.Dataset(
            inputs = [nt.ones((128,128))*i for i in range(10)],
            outputs = [i for i in range(10)]
        )
        self.assertEqual(len(dataset), 5)
        
        x, y = dataset[0]
        
        # test repr
        r = dataset.__repr__()
        
    def test_3d(self):
        dataset = datasets.CSVDataset(
            path=os.path.join(self.tmp_dir, 'participants.csv'),
            x={'images': 'filenames_3d'},
            y={'column': 'age'}
        )
        self.assertTrue(len(dataset.x) == 5)
        self.assertTrue(len(dataset.y) == 5)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)


if __name__ == '__main__':
    run_tests()
