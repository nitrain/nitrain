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
    
    def test_memory(self):
        dataset = nt.Dataset(
            inputs = [nti.ones((128,128))*i for i in range(10)],
            outputs = [i for i in range(10)]
        )
        self.assertEqual(len(dataset), 10)
        
        x, y = dataset[4]
        
        self.assertEqual(y, 4)
        self.assertTrue(nti.is_image(x))
        self.assertEqual(x.mean(), 4)
        
        # test repr
        r = dataset.__repr__()
        
    def test_multiple_memory(self):
        x = [nti.example('r16') for _ in range(10)]
        y = list(range(10))
        dataset = nt.Dataset([x, x], y)
        xx, yy = dataset[0]
        self.assertEqual(len(xx), 2)
        self.assertEqual(xx[0].shape, (256,256))
        self.assertEqual(xx[1].shape, (256,256))
        self.assertEqual(yy, 0)
        
        xx, yy = dataset[0:3]
        self.assertEqual(len(xx), 3)
        self.assertEqual(xx[0][0].shape, (256,256))
        self.assertEqual(xx[0][1].shape, (256,256))
        self.assertEqual(yy, [0,1,2])
        
    def test_memory_dict_inputs(self):
        dataset = nt.Dataset(
            inputs={'x':readers.ImageReader([nti.example('mni') for _ in range(10)]),
                    'y':readers.ImageReader([nti.example('mni') for _ in range(10)])},
            outputs=readers.ImageReader([nti.example('mni') for _ in range(10)])
        )
        x, y = dataset[0]
        self.assertEqual(len(x), 2)
        
        x, y = dataset[:3]
        self.assertEqual(len(x), 3)
        self.assertEqual(len(x[0]), 2)
    
    def test_memory_dict_inputs_with_transform(self):
        dataset = nt.Dataset(
            inputs={'x':readers.ImageReader([nti.example('mni') for _ in range(10)]),
                    'y':readers.ImageReader([nti.example('mni') for _ in range(10)])},
            outputs=readers.ImageReader([nti.example('mni') for _ in range(10)]),
            transforms={
                'y': [tx.Astype('uint8')]
            }
        )
        x, y = dataset[0]
        self.assertEqual(len(x), 2)
        
        x, y = dataset[:3]
        self.assertEqual(len(x), 3)
        self.assertEqual(len(x[0]), 2)
        
    def test_memory_array(self):
        dataset = nt.Dataset(
            inputs = [nti.ones((128,128))*i for i in range(10)],
            outputs = np.array([i for i in range(10)])
        )
        self.assertEqual(len(dataset), 10)
        
        x, y = dataset[4]
        
        self.assertEqual(y, 4)
        self.assertTrue(nti.is_image(x))
        self.assertEqual(x.mean(), 4)
        
        # test repr
        r = dataset.__repr__()
        
    def test_memory_double_inputs(self):
        dataset = nt.Dataset(
            inputs = [readers.ImageReader([nti.ones((128,128))*i for i in range(10)]),
                      readers.ImageReader([nti.ones((128,128))*i*2 for i in range(10)])],
            outputs = [i for i in range(10)]
        )
        self.assertEqual(len(dataset), 10)
        
        x, y = dataset[4]
        
        self.assertTrue(len(x), 2)
        self.assertTrue(nti.is_image(x[0]))
        self.assertTrue(nti.is_image(x[1]))
        self.assertEqual(x[0].mean(), 4)
        self.assertEqual(x[1].mean(), 8)
        self.assertEqual(y, 4)
        
        # test split
        ds_train, ds_test = dataset.split(0.8)
        self.assertEqual(len(ds_train), 8)
        self.assertEqual(len(ds_test), 2)
        
        x, y = ds_train[0]
        x2, y2 = ds_test[0]
        self.assertEqual(x[0].mean(), 0)
        self.assertEqual(x2[0].mean(), 8)
        self.assertEqual(y, 0)
        self.assertEqual(y2, 8)

class TestClass_CSVDataset(unittest.TestCase):
    def setUp(self):
        # set up directory
        tmp_dir = mkdtemp()
        img2d = nti.load(nti.example_data('r16'))
        img3d = nti.load(nti.example_data('mni'))
        
        filenames_2d = []
        filenames_3d = []
        for i in range(5):
            sub_dir = os.path.join(tmp_dir, f'sub_{i}')
            os.mkdir(sub_dir)
            filepath_2d = os.path.join(sub_dir, 'img2d.nii.gz')
            filenames_2d.append(filepath_2d)
            nti.save(img2d, filepath_2d)
            
            filepath_3d = os.path.join(sub_dir, 'img3d.nii.gz')
            filenames_3d.append(filepath_3d)
            nti.save(img3d, filepath_3d)
        
        # write csv file
        ids = [f'sub_{i}' for i in range(5)]
        age = [i + 50 for i in range(5)]
        df = pd.DataFrame({'sub_id': ids, 'age': age, 
                           'filenames_2d': filenames_2d,
                           'filenames_3d': filenames_3d})
        df.to_csv(os.path.join(tmp_dir, 'participants.csv'), index=False)
        
        self.tmp_dir = tmp_dir
         
    def tearDown(self):
       shutil.rmtree(self.tmp_dir)
    
    def test_2d(self):
        tmp_dir = self.tmp_dir
        dataset = nt.Dataset(
            inputs=readers.ColumnReader(base_file='participants.csv',
                                        column='filenames_2d',
                                        is_image=True),
            outputs=readers.ColumnReader(base_file='participants.csv',
                                         column='age'),
            base_dir=tmp_dir
        )
        self.assertEqual(len(dataset), 5)
        self.assertTrue(dataset.inputs.values[0].endswith('.nii.gz'))
        self.assertEqual(dataset.outputs.values[2], 52)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)
        self.assertTrue(nti.is_image(x[0]))
        self.assertEqual(y, [50, 51])
        
        # test repr
        r = dataset.__repr__()
        
    def test_3d(self):
        tmp_dir = self.tmp_dir
        dataset = nt.Dataset(
            inputs=readers.ColumnReader(column='filenames_3d', is_image=True),
            outputs=readers.ColumnReader(column='age'),
            base_file=os.path.join(tmp_dir, 'participants.csv')
        )
        self.assertEqual(len(dataset), 5)
        self.assertTrue(dataset.inputs.values[0].endswith('.nii.gz'))
        self.assertEqual(dataset.outputs.values[2], 52)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)
        self.assertTrue(nti.is_image(x[0]))
        self.assertEqual(y, [50, 51])

    def test_missing_file(self):
        with self.assertRaises(Exception):
            # file arg is needed somewhere
            dataset = nt.Dataset(
                inputs=readers.ColumnReader(column='filenames_3d', is_image=True),
                outputs=readers.ColumnReader(column='age')
            )

class TestClass_FolderDataset(unittest.TestCase):
    def setUp(self):
        # set up directory
        tmp_dir = mkdtemp()
        img2d = nti.load(nti.example_data('r16'))
        img3d = nti.load(nti.example_data('mni'))
        for i in range(5):
            sub_dir = os.path.join(tmp_dir, f'sub_{i}')
            os.mkdir(sub_dir)
            nti.save(img2d, os.path.join(sub_dir, 'img2d.nii.gz'))
            nti.save(img3d, os.path.join(sub_dir, 'img3d.nii.gz'))
        
        # write csv file
        ids = [f'sub_{i}' for i in range(5)]
        age = [i + 50 for i in range(5)]
        df = pd.DataFrame({'sub_id': ids, 'age': age})
        df.to_csv(os.path.join(tmp_dir, 'participants.csv'), index=False)
        
        self.tmp_dir = tmp_dir
        
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
    
    def test_2d(self):
        tmp_dir = self.tmp_dir
        dataset = nt.Dataset(
            inputs=readers.PatternReader('*/img2d.nii.gz'),
            outputs=readers.ColumnReader('age'),
            base_dir=tmp_dir,
            base_file=os.path.join(tmp_dir, 'participants.csv')   
        )
        self.assertEqual(len(dataset), 5)
        self.assertTrue(dataset.inputs.values[0].endswith('.nii.gz'))
        self.assertEqual(dataset.outputs.values[2], 52)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)
        self.assertTrue(nti.is_image(x[0]))
        self.assertEqual(y, [50, 51])
        
        # test split
        ds_train, ds_test = dataset.split(0.8)
        self.assertTrue(len(ds_train) > len(ds_test))
        
        # test repr
        r = dataset.__repr__()

    def test_2d_split(self):
        tmp_dir = self.tmp_dir
        dataset = nt.Dataset(
            inputs=readers.PatternReader('*/img2d.nii.gz'),
            outputs=readers.ColumnReader('age'),
            base_dir=tmp_dir,
            base_file=os.path.join(tmp_dir, 'participants.csv')   
        )
        
        ds_train, ds_test = dataset.split(0.8)
        self.assertTrue(len(ds_train) > len(ds_test))
        
        # test repr
        r = dataset.__repr__()
        
    def test_double_image_input(self):
        tmp_dir = self.tmp_dir
        dataset = nt.Dataset(
            inputs=[readers.PatternReader('*/img2d.nii.gz'),
                    readers.PatternReader('*/img3d.nii.gz')],
            outputs=readers.ColumnReader('age'),
            base_dir=tmp_dir,
            base_file=os.path.join(tmp_dir, 'participants.csv')   
        )
        self.assertEqual(len(dataset), 5)
        self.assertEqual(len(dataset.inputs.values), 5)
        self.assertEqual(len(dataset.inputs.values[0]), 2)
        self.assertTrue(dataset.inputs.values[0][0].endswith('.nii.gz'))
        self.assertEqual(dataset.outputs.values[2], 52)
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)
        self.assertTrue(len(x[0]) == 2)
        self.assertTrue(nti.is_image(x[0][0]))
        self.assertTrue(nti.is_image(x[0][1]))
        self.assertEqual(y, [50, 51])

    def test_2d_image_to_3d_image(self):
        tmp_dir = self.tmp_dir
        dataset = nt.Dataset(
            inputs=readers.PatternReader('*/img2d.nii.gz'),
            outputs=readers.PatternReader('*/img3d.nii.gz'),
            base_dir=tmp_dir
        )
        self.assertEqual(len(dataset), 5)
        self.assertTrue(dataset.inputs.values[0].endswith('.nii.gz'))
        self.assertTrue(dataset.outputs.values[0].endswith('.nii.gz'))
        
        x, y = dataset[:2]
        self.assertTrue(len(x) == 2)
        self.assertTrue(len(y) == 2)
        self.assertTrue(nti.is_image(x[0]))
        self.assertEqual(x[0].dimension, 2)
        self.assertEqual(y[0].dimension, 3)

class TestFunction_split(unittest.TestCase):
    
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_pattern_compose(self):
        base_dir = nt.fetch_data('example-01')

        dataset = nt.Dataset(inputs=[readers.PatternReader('*/img3d.nii.gz'),
                                     readers.PatternReader('*/img3d.nii.gz')],
                            outputs=readers.PatternReader('*/img3d_100.nii.gz'),
                            base_dir=base_dir)

        data_train, data_test = dataset.split(0.8)

        self.assertEqual(len(data_train), 8)
        self.assertEqual(len(data_test), 2)
        
        x,y=data_train[0]
        self.assertEqual(x[0].mean(), 1)
        self.assertEqual(y.mean(), 101)
        
        x2,y2=data_test[0]
        self.assertEqual(x2[0].mean(), 9)
        self.assertEqual(y2.mean(), 109)

class TestOther_Bugs(unittest.TestCase):
    def setUp(self):
        pass
        
    def tearDown(self):
        pass
    
    def test_multi_input_dict_transform(self):
        dataset = nt.Dataset(
            inputs={'x':readers.ImageReader([nti.example('mni') for _ in range(10)]),
                    'y':readers.ImageReader([nti.example('mni') for _ in range(10)])},
            outputs=readers.ImageReader([nti.example('mni') for _ in range(10)]),
            transforms={
                'x': [tx.Astype('uint8')]
            }
        )
        x, y = dataset[0]
        
        self.assertEqual(x[0].dtype, 'uint8')
        self.assertEqual(x[1].dtype, 'float32')
        self.assertEqual(y.dtype, 'float32')

        x, y = dataset[:2]
        self.assertEqual(x[0][0].dtype, 'uint8')
        self.assertEqual(x[0][1].dtype, 'float32')
        self.assertEqual(x[1][0].dtype, 'uint8')
        self.assertEqual(x[1][1].dtype, 'float32')
        
    def test_multi_input_dict_transform_no_list(self):
        dataset = nt.Dataset(
            inputs={'x':readers.ImageReader([nti.example('mni') for _ in range(10)]),
                    'y':readers.ImageReader([nti.example('mni') for _ in range(10)])},
            outputs=readers.ImageReader([nti.example('mni') for _ in range(10)]),
            transforms={
                'x': tx.Astype('uint8')
            }
        )
        x, y = dataset[0]
        
        self.assertEqual(x[0].dtype, 'uint8')
        self.assertEqual(x[1].dtype, 'float32')
        self.assertEqual(y.dtype, 'float32')
        
        x, y = dataset[:2]
        self.assertEqual(x[0][0].dtype, 'uint8')
        self.assertEqual(x[0][1].dtype, 'float32')
        self.assertEqual(x[1][0].dtype, 'uint8')
        self.assertEqual(x[1][1].dtype, 'float32')
    
    def test_no_base(self):
        base_dir = nt.fetch_data('example-01')
        
        
        dataset = nt.Dataset(inputs=readers.PatternReader('*/img3d.nii.gz',
                                                          base_dir=base_dir),
                            outputs=readers.PatternReader('*/img3d_seg.nii.gz',
                                                          base_dir=base_dir))
        
        x, y = dataset[0]
        self.assertEqual(x.shape, (30,40,50))

    def test_ids(self):
        base_dir = nt.fetch_data('example-01')
        
        dataset = nt.Dataset(inputs=readers.PatternReader('{id}/img3d.nii.gz'),
                            outputs=readers.PatternReader('{id}/img3d_seg.nii.gz'),
                            base_dir=base_dir)
        
        x, y = dataset[0]
        self.assertEqual(x.shape, (30,40,50))
            
    def test_exclude(self):
        base_dir = nt.fetch_data('example-01')
        
        dataset = nt.Dataset(inputs=readers.PatternReader('*/img3d.nii.gz',
                                                          exclude='sub_5/*'),
                            outputs=readers.PatternReader('*/img3d_seg.nii.gz',
                                                          exclude='sub_5/*'),
                            base_dir=base_dir)
        
        self.assertEqual(dataset.inputs.values, 9)
        
    def test_non_existent_files(self):
        base_dir = nt.fetch_data('example-01')
            
        with self.assertRaises(Exception):
            dataset = nt.Dataset(inputs=readers.PatternReader('*/img3d232.nii.gz'),
                                outputs=readers.PatternReader('*/img323d_seg.nii.gz'),
                                base_dir=base_dir)
            
        with self.assertRaises(Exception):
            dataset = nt.Dataset(inputs=readers.PatternReader('*/img3d.nii.gz'),
                                outputs=readers.ColumnReader('age', 'nonexist.csv'),
                                base_dir=base_dir)
            
if __name__ == '__main__':
    run_tests()
