# An example of how to download a dataset

import os
import bids
import nibabel
import datalad.api as dl
import numpy as np
import pandas as pd
from nitrain import utils, data

ds = utils.fetch_datalad('ds004711')

dataset = data.FileDataset(path=ds.path, 
                           layout='bids',
                           X_config = {'extension': 'nii.gz', 'datatype': 'anat', 'suffix': 'T1w'},
                           y_config = {'filename': 'participants.tsv', 'column': 'age'})

X, y = dataset.fetch_data(n=10)

dataset2 = data.Dataset(X, y)



