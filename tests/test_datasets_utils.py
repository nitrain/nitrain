import os
import unittest

from tempfile import mkdtemp
import shutil
import pandas as pd

import numpy as np
import ntimage as nti
import nitrain as nt
from nitrain import readers, transforms as tx

from main import run_tests

class TestFile_Utils(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = mkdtemp()
         
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
    
    def test_fetch_data_openneuro(self):
        # no dir
        #ds = nt.fetch_data('openneuro/ds004711', self.tmp_dir)
        
        # specific dir
        ds = nt.fetch_data('openneuro/ds004711', self.tmp_dir)
        self.assertEqual(ds, os.path.join(self.tmp_dir, 'openneuro/ds004711'))
    
    def test_fetch_data_example(self):
        ds = nt.fetch_data('example/dataset-01', self.tmp_dir)
    


if __name__ == '__main__':
    run_tests()
