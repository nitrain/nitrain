
from nitrain import datasets

dataset = datasets.GoogleCloudDataset(bucket='ants-dev',
                                         base_dir='datasets/nick-2/ds004711/', 
                                         x={'pattern': '*/anat/*_T1w.nii.gz', 'exclude': '**run-02*'},
                                         y={'file': 'participants.tsv', 'column': 'age'},
                                         credentials='/Users/ni5875cu/Desktop/ants.dev/engine/deep-dynamics-415608-4046316ec2f1.json')
x, y = dataset[0]