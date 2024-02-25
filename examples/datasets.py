# An example of how to download a dataset

import os
import bids
import nibabel
import datalad.api as dl
import numpy as np
import pandas as pd
from nitrain import utils, data

download = utils.fetch_datalad('ds004711')

# use a Datalad/BIDS folder -> create a FolderDataset 
ds = data.FolderDataset(path = download.path, 
                        layout = 'bids',
                        x_config = {'suffix': 'T1w', 'run': [None, '01']},
                        y_config = {'column': 'age'})

# filter data based on participants metadata file
ds_under40 = ds.filter('age < 40')
ds_males = ds.filter('gender == "male"')

# access image filenames via index (note this does not read in data)
x, y = ds[:5]

# load images into memory
x, y = ds.load(n=7)

## save filenames to csv file -> create a CSVDataset
#ds_csv = data.CSVDataset('dataset.csv')

## use in-memory images -> create a MemoryDataset
ds_memory = data.MemoryDataset(x, y)

loader = data.DatasetLoader(ds_memory, batch_size = 2)

# loop through all batches (one epoch)
for x, y in loader:
    print(y)

