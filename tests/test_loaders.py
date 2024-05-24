import os
import unittest
from main import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants
import nitrain as nt
from nitrain import samplers, readers, transforms as tx
from nitrain.readers import ImageReader
from nitrain.samplers import SliceSampler
from nitrain.loaders.loader import record_generator

class TestClass_DatasetLoader(unittest.TestCase):
    def setUp(self):
        img2d = ants.image_read(ants.get_data('r16'))
        img3d = ants.image_read(ants.get_data('mni'))
        
        x = [img2d for _ in range(10)]
        y = list(range(10))
        
        dataset_2d = nt.Dataset(x, y)
        
        x = [img3d for _ in range(10)]
        y = list(range(10))
        
        dataset_3d = nt.Dataset(x, y)
        
        self.dataset_2d = dataset_2d
        self.dataset_3d = dataset_3d
        
    def tearDown(self):
        pass
    
    def test_2d(self):
        import ants
        import nitrain as nt
        img2d = ants.image_read(ants.get_data('r16'))
        img3d = ants.image_read(ants.get_data('mni'))
        
        x = [img2d for _ in range(10)]
        y = list(range(10))
        
        dataset_2d = nt.Dataset(x, y)
        loader = nt.Loader(dataset_2d, images_per_batch=4)
        xb, yb = next(iter(loader))
        self.assertEqual(xb.shape, (4, 256, 256, 1))
        self.assertEqual(yb.shape, (4,))
        
        loader = nt.Loader(dataset_2d, images_per_batch=4, channel_axis=None)
        xb, yb = next(iter(loader))
        self.assertEqual(xb.shape, (4, 256, 256))
        self.assertEqual(yb.shape, (4,))
        
        # test repr
        rep = loader.__repr__()
    
    def test_to_keras(self):
        loader = nt.Loader(self.dataset_2d, images_per_batch=4)
        keras_loader = loader.to_keras()
        x_batch, y_batch = next(iter(keras_loader))
        self.assertEqual(x_batch.shape, (4, 256, 256, 1))
        
        gen = record_generator(loader)
        xb,yb = next(iter(gen))
        
    def test_keras_multi(self):
        img2d = ants.image_read(ants.get_data('r16'))
        x = [img2d for _ in range(10)]
        y = list(range(10))

        dataset_2d = nt.Dataset([x,x], y)

        loader = nt.Loader(dataset_2d, images_per_batch=4)
        keras_loader = loader.to_keras()
        xb, yb = next(iter(keras_loader))
        
        self.assertEqual(len(xb), 2)
        self.assertEqual(xb[0].shape, (4,256,256,1))
        self.assertEqual(xb[1].shape, (4,256,256,1))
        self.assertEqual(yb.shape, (4))
        
        gen = record_generator(loader)
        xb,yb = next(iter(gen))
        
    def test_3d(self):
        loader = nt.Loader(self.dataset_3d, images_per_batch=4)
        
        self.assertTrue(len(loader) > 0)
        
        rep = loader.__repr__()
        
        x_batch, y_batch = next(iter(loader))
        self.assertEqual(x_batch.shape,  (4, 182, 218, 182, 1))
        
    def test_3d_no_expand(self):
        loader = nt.Loader(self.dataset_3d, images_per_batch=4,
                           channel_axis=None)
        
        x_batch, y_batch = next(iter(loader))
        
        loader2 = loader.to_keras()
        x_batch, y_batch = next(iter(loader2))
        self.assertTrue(x_batch.shape == (4, 182, 218, 182))
        
        gen = record_generator(loader)
        xb,yb = next(iter(gen))
    
    def test_image_to_image(self):
        img = ants.image_read(ants.get_data('r16'))
        x = [img for _ in range(10)]
        dataset = nt.Dataset(x, x)
        loader = nt.Loader(dataset, images_per_batch=4)

        x_batch, y_batch = next(iter(loader))
        
        self.assertTrue(x_batch.shape == (4, 256, 256, 1))
        self.assertTrue(y_batch.shape == (4, 256, 256, 1))
        
        loader2 = loader.to_keras()
        x_batch, y_batch = next(iter(loader2))
        
        self.assertTrue(x_batch.shape == (4, 256, 256, 1))
        self.assertTrue(y_batch.shape == (4, 256, 256, 1))
        
        gen = record_generator(loader)
        xb,yb = next(iter(gen))

    def test_multi_image_to_image(self):
        import ants
        import nitrain as nt
        img = ants.from_numpy(np.zeros((256,256)))
        dataset = nt.Dataset([[img for _ in range(10)], 
                              [img for _ in range(10)]],
                             [img for _ in range(10)])
        loader = nt.Loader(dataset, images_per_batch=4)

        xb, yb = next(iter(loader))
        self.assertTrue(len(xb) == 2)
        self.assertTrue(xb[0].shape == (4, 256, 256, 1))
        self.assertTrue(xb[1].shape == (4, 256, 256, 1))
        self.assertTrue(yb.shape == (4, 256, 256, 1))
        
        loader2 = loader.to_keras()
        x_batch, y_batch = next(iter(loader2))
        self.assertTrue(len(x_batch) == 2)
        self.assertTrue(tuple(x_batch[0].shape) == (4, 256, 256, 1))
        self.assertTrue(tuple(x_batch[1].shape) == (4, 256, 256, 1))
        self.assertTrue(tuple(y_batch.shape) == (4, 256, 256, 1))
        
        gen = record_generator(loader)
        xb,yb = next(iter(gen))
    
    def test_image_to_image_with_slice_sampler(self):
        img = ants.image_read(ants.get_data('mni'))
        x = [img for _ in range(10)]
        dataset = nt.Dataset(x, x)
        loader = nt.Loader(dataset, 
                           images_per_batch=4,
                           sampler=samplers.SliceSampler(batch_size=12, axis=0))

        x_batch, y_batch = next(iter(loader))
        self.assertEqual(x_batch.shape, (12, 218, 182, 1))
        self.assertEqual(y_batch.shape, (12, 218, 182, 1))
        
        loader2 = loader.to_keras()
        x_batch, y_batch = next(iter(loader2))
        self.assertEqual(tuple(x_batch.shape), (12, 218, 182, 1))
        self.assertEqual(tuple(y_batch.shape), (12, 218, 182, 1))
        
    def test_multiple_image_slice(self):
        base_dir = nt.fetch_data('example-01')

        dataset = nt.Dataset(inputs={'a': readers.ImageReader('*/img3d.nii.gz'),
                                    'b': readers.ImageReader('*/img3d_100.nii.gz')},
                            outputs=readers.ImageReader('*/img3d_seg.nii.gz'),
                            base_dir=base_dir)

        loader = nt.Loader(dataset,
                           images_per_batch=1,
                           sampler=samplers.SliceSampler(batch_size=50, axis=-1))

        for i, (xbatch, ybatch) in enumerate(loader):
                self.assertEqual(xbatch[0].mean(), i+1)
                self.assertEqual(xbatch[1].mean(), i+1+100)

                
    def test_multiple_image_slice_after_split(self):
        base_dir = nt.fetch_data('example-01')

        dataset = nt.Dataset(inputs={'a': readers.ImageReader('*/img3d.nii.gz'),
                                    'b': readers.ImageReader('*/img3d_100.nii.gz')},
                            outputs=readers.ImageReader('*/img3d_seg.nii.gz'),
                            base_dir=base_dir)
        
        ds_train, ds_test = dataset.split(0.8, random=False)
        
        loader = nt.Loader(ds_train,
                           images_per_batch=1,
                           sampler=samplers.SliceSampler(batch_size=50, axis=-1))

        for i, (xbatch, ybatch) in enumerate(loader):
                self.assertEqual(xbatch[0].mean(), i+1)
                self.assertEqual(xbatch[1].mean(), i+1+100)
                
    def test_transforms(self):
        import nitrain as nt
        import ants
        from nitrain.readers import ImageReader
        from nitrain.samplers import SliceSampler
        from nitrain import transforms as tx
        base_dir = nt.fetch_data('example-01')

        dataset = nt.Dataset(inputs=[ImageReader('*/img3d.nii.gz'),
                                     ImageReader('*/img3d.nii.gz')],
                            outputs=ImageReader('*/img3d_100.nii.gz'),
                            base_dir=base_dir)

        loader = nt.Loader(dataset,
                           images_per_batch=2,
                           transforms={
                               ('inputs', 'outputs'): tx.Resample((48,48,48))
                            },
                           sampler=SliceSampler(batch_size=20, axis=-1))
        xb,yb = next(iter(loader))
        
        self.assertEqual(xb[0].shape, (20,48,48,1))
        self.assertEqual(xb[1].shape, (20,48,48,1))
        self.assertEqual(yb.shape, (20,48,48,1))
        
        loader = nt.Loader(dataset,
                           images_per_batch=2,
                           transforms={
                               'inputs': tx.Resample((48,48,48))
                            },
                           sampler=SliceSampler(batch_size=20, axis=-1))
        xb,yb = next(iter(loader))
        
        self.assertEqual(xb[0].shape, (20,48,48,1))
        self.assertEqual(xb[1].shape, (20,48,48,1))
        self.assertEqual(yb.shape, (20,30,40,1))
        
    def test_multiclass_segmentation_no_expand_dims(self):
        base_dir = nt.fetch_data('example-01')

        dataset = nt.Dataset(inputs=ImageReader('*/img3d.nii.gz'),
                            outputs=ImageReader('*/img3d_multiseg.nii.gz'),
                            transforms={
                                    ('inputs','outputs'): tx.Resample((40,40,40)),
                                    'inputs': tx.AddChannel(),
                                    'outputs': tx.LabelsToChannels()
                            },
                            base_dir=base_dir)

        x,y = dataset[0]
        
        data_train, data_test = dataset.split(0.8, random=False)

        loader = nt.Loader(data_train,
                           images_per_batch=4,
                           shuffle=True,
                           channel_axis=None,
                           sampler=SliceSampler(batch_size=20, axis=2))
        
        xb, yb = next(iter(loader))
        
        self.assertEqual(xb.shape, (20,40,40,1))
        self.assertEqual(yb.shape, (20,40,40,2))

    
if __name__ == '__main__':
    run_tests()
