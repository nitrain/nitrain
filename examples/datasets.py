# An example of how to download a dataset

import os
import bids
import nibabel
import datalad.api as dl
import numpy as np
import pandas as pd
from nitrain import utils, data

download = utils.fetch_datalad('ds004711')
        
ds = data.FolderDataset(path = download.path, 
                         layout = 'bids',
                         x_config = {'suffix': 'T1w', 'run': [None, '01']},
                         y_config = {'column': 'age'})

# filter data based on participants metadata file
ds_filtered = ds.filter('age < 18')
ds_females = ds.filter('gender == "female"')

# load some images into memory
x, y = ds.load(n=5)

# access data via index (note this does not read in data)
x, y = ds[0]



