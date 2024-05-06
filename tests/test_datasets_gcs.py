import os
import unittest

from tempfile import NamedTemporaryFile
import base64
import json

import nitrain as nt
from nitrain import readers, transforms as tx
        
from main import run_tests

class TestClass_GoogleCloudDataset(unittest.TestCase):
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
    
    def test_gcs_basic(self):
        d = nt.GoogleCloudDataset(
            inputs=readers.ImageReader('sub-*/anat/*_T1w.nii.gz'),
            outputs=readers.ColumnReader('age', 'participants.tsv'),
            base_dir='datasets/nick-2/ds004711',
            bucket='ants-dev',
            credentials = self.credentials.name
        )
        
        self.assertTrue(len(d.inputs.values) > 0)
        
    def test_gcs_basic_base(self):
        d = nt.GoogleCloudDataset(
            inputs=readers.ImageReader('sub-*/anat/*_T1w.nii.gz',
                                         base_dir='datasets/nick-2/ds004711'),
            outputs=readers.ColumnReader('age',
                                         base_file='datasets/nick-2/ds004711/participants.tsv'),
            bucket='ants-dev',
            credentials = self.credentials.name
        )
        
        self.assertTrue(len(d.inputs.values) > 0)

    def test_gcs_compose(self):
        d = nt.GoogleCloudDataset(
            inputs=[readers.ImageReader('sub-*/anat/*_T1w.nii.gz'),
                    readers.ImageReader('sub-*/anat/*_T1w.nii.gz')],
            outputs=readers.ColumnReader('age', 'participants.tsv'),
            base_dir='datasets/nick-2/ds004711',
            bucket='ants-dev',
            credentials = self.credentials.name
        )
        
        self.assertEqual(len(d.inputs.values[0]), 2)
        self.assertTrue(len(d.inputs.values) > 0)
        
if __name__ == '__main__':
    run_tests()
