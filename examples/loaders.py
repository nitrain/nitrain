# An example of how to download a dataset

import os
import bids
import nibabel
import datalad.api as dl
import numpy as np
import pandas as pd
from nitrain import utils, datasets, loaders, transforms as tx

download = utils.fetch_datalad('ds004711')

# use a Datalad/BIDS folder -> create a FolderDataset 
ds = datasets.FolderDataset(path = download.path, 
                        layout = 'bids',
                        x_config = {'suffix': 'T1w', 
                                    'scope': 'derivatives',
                                    'desc': 'precompute'},
                        y_config = {'column': 'age'})
# load in some data
x, y = ds[:5]

## create a memory dataset for 
ds_memory = datasets.MemoryDataset(x, y)

loader = datasets.DatasetLoader(ds_memory,
                            batch_size=2)

# loop through each x, y pair for one epoch
for x, y in loader:
    print(y)
    
loader = loaders.DatasetLoader(ds_memory,
                            batch_size=2)