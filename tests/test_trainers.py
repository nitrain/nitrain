import os
import unittest
from main import run_tests

from tempfile import mktemp

import numpy as np
import numpy.testing as nptest

import ants
from nitrain import datasets, loaders, models, trainers


class TestClass_ModelTrainer(unittest.TestCase):
    def setUp(self):
        img = ants.image_read(ants.get_data('r16')).resample_image((4,4))
        x = [img for _ in range(6)]
        y = list(range(6))
        dataset = datasets.MemoryDataset(x, y)
        loader = loaders.DatasetLoader(dataset, batch_size=4)
        arch_fn = models.fetch_architecture('vgg', dim=2)
        model = arch_fn(input_image_size=(64,64,1), 
                        number_of_classification_labels=1,
                        mode='regression')
        self.loader = loader
        self.model = model

    def tearDown(self):
        pass
    
    def test_trainer(self):
        trainer = trainers.ModelTrainer(self.model, task='regression')
        trainer.fit(self.loader, epochs=2)
        res = trainer.evaluate(self.loader)
        pred = trainer.predict(self.loader)
        s = trainer.summary()

        
if __name__ == '__main__':
    run_tests()
