import os
import unittest
from main import run_tests

import tempfile

import numpy as np
import numpy.testing as nptest

import ntimage as nti
import nitrain as nt
from nitrain import transforms as tx
from nitrain.readers import ImageReader
from nitrain.samplers import SliceSampler

class TestClass_OneInput_OneOutput(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_binary_segmentation(self):
        base_dir = nt.fetch_data('example-01')

        dataset = nt.Dataset(inputs=ImageReader('*/img3d.nii.gz'),
                            outputs=ImageReader('*/img3d_seg.nii.gz'),
                            transforms={
                                    ('inputs','outputs'): tx.Resample((40,40,40))
                            },
                            base_dir=base_dir)

        x,y = dataset[0]

        data_train, data_test = dataset.split(0.8)

        loader = nt.Loader(data_train,
                        images_per_batch=4,
                        shuffle=True,
                        sampler=SliceSampler(batch_size=20, axis=-1))

        arch_fn = nt.fetch_architecture('unet', dim=2)
        model = arch_fn(x.shape[:-1]+(1,),
                        number_of_outputs=1,
                        number_of_layers=4,
                        number_of_filters_at_base_layer=16,
                        mode='sigmoid')

        # train
        trainer = nt.Trainer(model, task='segmentation')
        trainer.fit(loader, epochs=2)
        
        # evaluate on test data
        test_loader = loader.copy(data_test)
        trainer.evaluate(test_loader)
        
        # inference on test data
        predictor = nt.Predictor(model, 
                                 task='segmentation',
                                 sampler=SliceSampler(axis=-1))
        y_pred = predictor.predict(data_test)
    
    def test_multiclass_segmentation(self):
        import ntimage as nti
        import nitrain as nt
        from nitrain import transforms as tx
        from nitrain.readers import ImageReader
        from nitrain.samplers import SliceSampler
        base_dir = nt.fetch_data('example-01')

        dataset = nt.Dataset(inputs=ImageReader('*/img3d.nii.gz'),
                            outputs=ImageReader('*/img3d_multiseg.nii.gz'),
                            transforms={
                                    ('inputs','outputs'): tx.Resample((40,40,40)),
                                    'outputs': tx.ExpandLabels()
                            },
                            base_dir=base_dir)

        x,y = dataset[0]
        
       # data_train, data_test = dataset.split(0.8)

        #loader = nt.Loader(dataset,
        #                   images_per_batch=4)
        #
        #xb, yb = next(iter(loader))
#
        #arch_fn = nt.fetch_architecture('unet', dim=2)
        #model = arch_fn(x.shape[:-1]+(1,),
        #                number_of_outputs=2,
        #                number_of_layers=4,
        #                number_of_filters_at_base_layer=16,
        #                mode='classification')
#
        ## train
        #trainer = nt.Trainer(model, task='segmentation')
        #trainer.fit(loader, epochs=2)
        #
        ## evaluate on test data
        #test_loader = loader.copy(data_test)
        #trainer.evaluate(test_loader)
        #
        ## inference on test data
        #predictor = nt.Predictor(model, 
        #                         task='segmentation',
        #                         sampler=SliceSampler(axis=-1))
        #y_pred = predictor.predict(data_test)
    
    def test_image_regression(self):
        pass
    
    def test_scalar_regression(self):
        pass
    
    def test_binary_scalar_classification(self):
        pass
    
    def test_multiclass_scalar_classification(self):
        pass
        
if __name__ == '__main__':
    run_tests()
