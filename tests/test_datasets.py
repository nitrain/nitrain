import os
import unittest
from main import run_tests

from tempfile import mktemp, mkdtemp
import shutil

import pandas as pd
import numpy as np
import numpy.testing as nptest

import ants
from nitrain import datasets


class TestClass_MemoryDataset(unittest.TestCase):
    def setUp(self):
        self.img2d = ants.image_read(ants.get_data('r16'))
        self.img3d = ants.image_read(ants.get_data('mni'))

    def tearDown(self):
        pass
    
    def test_2d(self):
        x = [self.img2d for _ in range(10)]
        y = list(range(10))
        
        dataset = datasets.MemoryDataset(x, y)
        self.assertTrue(len(dataset.x) == 10)
        
    def test_3d(self):
        x = [self.img3d for _ in range(10)]
        y = list(range(10))
        
        dataset = datasets.MemoryDataset(x, y)
        self.assertTrue(len(dataset.x) == 10)


class TestClass_FolderDataset(unittest.TestCase):
    def setUp(self):
        # set up directory
        tmp_dir = mkdtemp()
        self.tmp_dir = tmp_dir
        img2d = ants.image_read(ants.get_data('r16'))
        img3d = ants.image_read(ants.get_data('mni'))
        for i in range(5):
            sub_dir = os.path.join(tmp_dir, f'sub_{i}')
            os.mkdir(sub_dir)
            ants.image_write(img2d, os.path.join(sub_dir, 'img2d.nii.gz'))
            ants.image_write(img3d, os.path.join(sub_dir, 'img3d.nii.gz'))
        
        # write csv file
        ids = [f'sub_{i}' for i in range(5)]
        age = [i + 50 for i in range(5)]
        df = pd.DataFrame({'sub_id': ids, 'age': age})
        df.to_csv(os.path.join(tmp_dir, 'participants.csv'), index=False)
        
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
    
    def test_2d(self):
        dataset = datasets.FolderDataset(
            base_dir=self.tmp_dir,
            x={'pattern': '*/img2d.nii.gz'},
            y={'file': 'participants.csv', 'column': 'age'}
        )
        self.assertTrue(len(dataset.x) == 5)
        self.assertTrue(len(dataset.y) == 5)
        
    def test_3d(self):
        dataset = datasets.FolderDataset(
            base_dir=self.tmp_dir,
            x={'pattern': '*/img3d.nii.gz'},
            y={'file': 'participants.csv', 'column': 'age'}
        )
        self.assertTrue(len(dataset.x) == 5)
        self.assertTrue(len(dataset.y) == 5)
        

if __name__ == '__main__':
    run_tests()
