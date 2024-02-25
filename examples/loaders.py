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

# access image filenames via index (note this does not read in data)
x, y = ds[:5]



# load images into memory
x, y = ds.load(n=7)


## use in-memory images -> create a MemoryDataset
ds_memory = data.MemoryDataset(x, y)

loader = data.DatasetLoader(ds_memory,
                            batch_size=2)


for x, y in loader:
    print(y)