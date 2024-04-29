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

class TestClass_Dataset(unittest.TestCase):
    def setUp(self):
        pass
         
    def tearDown(self):
        pass
    
    def test_gcs(self):
        d = nt.GCSDataset(
            inputs=readers.PatternReader('sub-*/anat/*_T1w.nii.gz'),
            outputs=readers.ColumnReader('participants.tsv', 'age'),
            base_dir='datasets/nick-2/ds004711',
            bucket='ants-dev',
            credentials = os.environ['GCP64']
        )
        
        self.assertTrue(len(d.inputs.values) > 0)

if __name__ == '__main__':
    run_tests()
