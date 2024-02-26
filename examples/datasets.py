# An example of how to download a dataset

import os
import bids
import nibabel
import datalad.api as dl
import numpy as np
import pandas as pd
import ants
from nitrain import utils, data

download = utils.fetch_datalad('ds004711')

def transform_fn(img):
    img = img.resample_image((4,4,4))
    #img = img.slice_image(2, 32)
    img = img.mask_image(ants.get_mask(img))
    return img

# use a Datalad/BIDS folder -> create a FolderDataset 
ds = data.FolderDataset(path = download.path, 
                        layout = 'bids',
                        x_config = {'suffix': 'T1w', 'run': [None, '01']},
                        y_config = {'column': 'age'},
                        x_transform = transform_fn)

# run the transform function on all input images and save results to derivatives/nitrain/
ds.precompute_transforms(desc='precompute')


# read in + transform the first three images via indexing
x, y = ds[0:3]

# filter data based on participants metadata file
ds_under22 = ds.filter('age < 22')
ds_males = ds.filter('gender == "male"')


## save filenames to csv file -> create a CSVDataset
#ds_csv = data.CSVDataset('dataset.csv')

## use in-memory images -> create a MemoryDataset
ds_memory = data.MemoryDataset(x, y)

