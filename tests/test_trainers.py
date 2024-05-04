import os
import unittest
from main import run_tests

import tempfile

import numpy as np
import numpy.testing as nptest

import ntimage as nti
import nitrain as nt


class TestClass_LocalTrainer(unittest.TestCase):
    def setUp(self):
        img = nti.load(nti.example_data('r16')).resample((4,4), use_spacing=True)
        x = [img for _ in range(6)]
        y = list(range(6))
        dataset = nt.Dataset(x, y)
        loader = nt.Loader(dataset, images_per_batch=4)
        arch_fn = nt.fetch_architecture('vgg', dim=2)
        model = arch_fn(input_image_size=(64,64,1), 
                        number_of_classification_labels=1,
                        mode='regression')
        self.loader = loader
        self.model = model

    def tearDown(self):
        pass
    
    def test_regression_keras(self):
        trainer = nt.Trainer(self.model, task='regression')
        trainer.fit(self.loader, epochs=2)
        
        trainer.evaluate(self.loader)
        trainer.predict(self.loader)
        trainer.summary()

        tmpfile = tempfile.NamedTemporaryFile(suffix='.keras')
        trainer.save(tmpfile.name)
        tmpfile.close()
        
        trainer.__repr__()
    
    def test_exceptions(self):
        with self.assertRaises(Exception):
            trainer = nt.Trainer(123, task='regression')
            
        with self.assertRaises(Exception):
            trainer = nt.Trainer(self.model, task='wrongtask')
        
        
if __name__ == '__main__':
    run_tests()
