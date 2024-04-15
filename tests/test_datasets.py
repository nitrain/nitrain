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

import ants
from nitrain import datasets, transforms as tx
from nitrain.datasets.configs import _infer_config


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
        
        # test repr
        r = dataset.__repr__()

    def test_2d_and_x_transforms(self):
        x = [self.img2d for _ in range(10)]
        y = list(range(10))
        
        dataset = datasets.MemoryDataset(x, y,
                                         x_transforms=[tx.Resample((69,69))])
        self.assertTrue(len(dataset.x) == 10)
        
        x, y = dataset[0]
        self.assertTrue(x.shape==(69,69))
        
    def test_2d_image_to_image_and_x_transforms(self):
        x = [self.img2d for _ in range(10)]
        y = [self.img2d for _ in range(10)]
        
        dataset = datasets.MemoryDataset(x, y,
                                         co_transforms=[tx.Resample((69,69))])
        self.assertTrue(len(dataset.x) == 10)
        
        x, y = dataset[0]
        self.assertTrue(x.shape==(69,69))
        self.assertTrue(y.shape==(69,69))
        
    def test_3d(self):
        x = [self.img3d for _ in range(10)]
        y = list(range(10))
        
        dataset = datasets.MemoryDataset(x, y)
        self.assertTrue(len(dataset.x) == 10)
        
    def test_from_array(self):
        dataset = datasets.MemoryDataset(
            np.random.normal(20,10,(5,50,50)),
            np.random.normal(20,10,5)
        )
        x, y = dataset[0]
        self.assertTrue(x.shape == (50,50))
        self.assertTrue(isinstance(y, float))

    def test_from_array_image_to_Image(self):
        dataset = datasets.MemoryDataset(
            np.random.normal(20,10,(5,50,50)),
            np.random.normal(20,10,(5,50,50)),
        )
        x, y = dataset[0]
        self.assertTrue(x.shape == (50,50))
        self.assertTrue(y.shape == (50,50))


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
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)
        
        # test repr
        r = dataset.__repr__()
        
    def test_double_image_input(self):
        dataset = datasets.FolderDataset(
            base_dir=self.tmp_dir,
            x=[{'pattern': '*/img2d.nii.gz'},{'pattern': '*/img2d.nii.gz'}],
            y={'file': 'participants.csv', 'column': 'age'}
        )
        self.assertTrue(len(dataset.x) == 5)
        self.assertTrue(len(dataset.x[0]) == 2)
        self.assertTrue(len(dataset.y) == 5)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)
        self.assertTrue(len(x[0]) == 2)

    def test_2d_image_to_image(self):
        dataset = datasets.FolderDataset(
            base_dir=self.tmp_dir,
            x={'pattern': '*/img2d.nii.gz'},
            y={'pattern': '*/img2d.nii.gz'}
        )
        self.assertTrue(len(dataset.x) == 5)
        self.assertTrue(len(dataset.y) == 5)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)

    def test_2d_image_to_image_and_x_transforms(self):
        dataset = datasets.FolderDataset(
            base_dir=self.tmp_dir,
            x={'pattern': '*/img2d.nii.gz'},
            y={'pattern': '*/img2d.nii.gz'},
            x_transforms=[tx.Resample((69,69))]
        )
        self.assertTrue(len(dataset.x) == 5)
        self.assertTrue(len(dataset.y) == 5)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)
        
        self.assertTrue(x[0].shape==(69,69))
        self.assertTrue(y[0].shape!=(69,69))
        
    def test_2d_image_to_image_and_co_transforms(self):
        dataset = datasets.FolderDataset(
            base_dir=self.tmp_dir,
            x={'pattern': '*/img2d.nii.gz'},
            y={'pattern': '*/img2d.nii.gz'},
            co_transforms=[tx.Resample((69,69))]
        )
        self.assertTrue(len(dataset.x) == 5)
        self.assertTrue(len(dataset.y) == 5)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)
        
        self.assertTrue(x[0].shape==(69,69))
        self.assertTrue(y[0].shape==(69,69))
        
    def test_3d(self):
        dataset = datasets.FolderDataset(
            base_dir=self.tmp_dir,
            x={'pattern': '*/img3d.nii.gz'},
            y={'file': 'participants.csv', 'column': 'age'}
        )
        self.assertTrue(len(dataset.x) == 5)
        self.assertTrue(len(dataset.y) == 5)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)


class TestClass_CSVDataset(unittest.TestCase):
    def setUp(self):
        # set up directory
        tmp_dir = mkdtemp()
        self.tmp_dir = tmp_dir
        img2d = ants.image_read(ants.get_data('r16'))
        img3d = ants.image_read(ants.get_data('mni'))
        
        filenames_2d = []
        filenames_3d = []
        for i in range(5):
            sub_dir = os.path.join(tmp_dir, f'sub_{i}')
            os.mkdir(sub_dir)
            filepath_2d = os.path.join(sub_dir, 'img2d.nii.gz')
            filenames_2d.append(filepath_2d)
            ants.image_write(img2d, filepath_2d)
            
            filepath_3d = os.path.join(sub_dir, 'img3d.nii.gz')
            filenames_3d.append(filepath_3d)
            ants.image_write(img3d, filepath_3d)
        
        # write csv file
        ids = [f'sub_{i}' for i in range(5)]
        age = [i + 50 for i in range(5)]
        df = pd.DataFrame({'sub_id': ids, 'age': age, 
                           'filenames_2d': filenames_2d,
                           'filenames_3d': filenames_3d})
        df.to_csv(os.path.join(tmp_dir, 'participants.csv'), index=False)
         
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
    
    def test_2d(self):
        dataset = datasets.CSVDataset(
            path=os.path.join(self.tmp_dir, 'participants.csv'),
            x={'images': 'filenames_2d'},
            y={'column': 'age'}
        )
        self.assertTrue(len(dataset.x) == 5)
        self.assertTrue(len(dataset.y) == 5)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)
        
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

class TestClass_BIDSDataset(unittest.TestCase):
    def setUp(self):
        # set up directory
        tmp_dir = mkdtemp()
        self.tmp_dir = tmp_dir
        img2d = ants.image_read(ants.get_data('r16'))
        img3d = ants.image_read(ants.get_data('mni'))
        
        for i in range(5):
            sub_dir = os.path.join(tmp_dir, f'sub-00{i}')
            os.makedirs(os.path.join(sub_dir, 'anat/'))
            
            filepath_2d = os.path.join(sub_dir, f'anat/sub-00{i}_T1w.nii.gz')
            ants.image_write(img2d, filepath_2d)
            
            filepath_3d = os.path.join(sub_dir, f'anat/sub-00{i}_T2w.nii.gz')
            ants.image_write(img3d, filepath_3d)
        
        # write csv file
        ids = [f'sub-00{i}' for i in range(5)]
        age = [i + 50 for i in range(5)]
        df = pd.DataFrame({'sub_id': ids, 'age': age})
        df.to_csv(os.path.join(tmp_dir, 'participants.tsv'), index=False, sep='\t')
        
        description = {'Name': 'Test Dataste', 
                       'BIDSVersion': 'v1.8.0 (2022-10-29)'}
        with open (os.path.join(tmp_dir, 'dataset_description.json'), 'w') as outfile:
            json.dump(description, outfile)
         
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
    
    def test_2d(self):
        dataset = datasets.BIDSDataset(self.tmp_dir, 
                                       x={'datatype': 'anat', 'suffix': 'T1w'},
                                       y={'file':'participants.tsv', 'column':'age'})
        self.assertTrue(len(dataset.x) == 5)
        self.assertTrue(len(dataset.y) == 5)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)
        
        # test repr
        r = dataset.__repr__()
        
    def test_3d(self):
        dataset = datasets.BIDSDataset(self.tmp_dir, 
                                       x={'datatype': 'anat', 'suffix': 'T2w'},
                                       y={'file':'participants.tsv', 'column':'age'})
        self.assertTrue(len(dataset.x) == 5)
        self.assertTrue(len(dataset.y) == 5)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)

class TestClass_GoogleCloudDataset(unittest.TestCase):
    # NOTE: GCP credentials are required for this test
    
    def setUp(self):
        base64_string = os.environ.get('GCP64')
        decodedBytes = base64.b64decode(base64_string)
        decodedStr = decodedBytes.decode("ascii") 
        object = json.loads(decodedStr)
        file = NamedTemporaryFile(suffix='.json')
        with open(file.name, 'w') as f:
            json.dump(object, f)
        self.credentials = file
        
    def tearDown(self):
        pass

    def test_gcp_lazy(self):
        dataset = datasets.GoogleCloudDataset(bucket='ants-dev',
                                              base_dir='datasets/nick-2/ds004711', 
                                              x={'pattern': '*/anat/*_T1w.nii.gz', 'exclude': '**run-02*'},
                                              y={'file': 'participants.tsv', 'column': 'age'},
                                              lazy=True)
        self.assertIsNone(dataset.x)
        
    def test_gcp(self):
        dataset = datasets.GoogleCloudDataset(bucket='ants-dev',
                                              base_dir='datasets/nick-2/ds004711', 
                                              x={'pattern': '*/anat/*_T1w.nii.gz', 'exclude': '**run-02*'},
                                              y={'file': 'participants.tsv', 'column': 'age'},
                                              credentials=self.credentials.name)
        self.assertEqual(len(dataset.x), 187)
        self.assertEqual(len(dataset.y), 187)
        
        x, y = dataset[0]
        self.assertEqual(x.shape, (256, 256, 192))
        
        
if __name__ == '__main__':
    run_tests()
