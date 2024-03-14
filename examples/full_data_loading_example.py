# This is a complete, self-contained example of training a brain-age model
# using the full data loading functionality from nitrain.
# This example demonstrates nitrain's `datasets`, `loaders`, `samplers`, and `transforms`,
# but it lets you use your own code to create and train the model.

import os
from nitrain import datasets, loaders, samplers, transforms as tx

### fetch a dataset
# The ds004711 from openneuro will be cloned using datalad, so you have to have datalad installed.
# this does not actually download any image files, so it should be quick
# images are only downloaded the first time they are used, so it should not take up much space
work_dir = os.path.expanduser('~/Desktop')
datasets.fetch_data('openneuro/ds004711', path=work_dir)

### create folder dataset
# inputs (x) will be the first T1 image for each subject (n = 187)
# The input files are selected using convenient bids entities (see pybids package), but if you didnt have a BIDS
# dataset then you can instead grab files uing glob patterns with a FolderDataset.
# outputs (y) will be the age values for the subjects taken from the participants.tsv file
# And when the T1 images (x) are first read into file, they will be skull-stripped and downsampled by 4x.
# The x_transforms passed to a dataset should be non-random, preprocessing steps that ideally
# would be done beforehand. And indeed it is possible to precompute the dataset transforms (not shown here).
dataset = datasets.BIDSDataset(base_dir=os.path.join(work_dir, 'openneuro/ds004711'),
                               x={'suffix': 'T1w', 'run': [None, '01']},
                               y={'file': 'participants.tsv', 'column': 'age'},
                               x_transforms=[tx.Resample((4,4,4), use_spacing=True),
                                             tx.BrainExtraction()])

### create sampler
# samplers are needed if you want to train your model on something other than the full image.
# This means you need samplers if you want to train on 1) 2D slices of 3D images, 2) 3D blocks
# 3D images, 3) 2D patches of 2D images, or 4) 2D patches of 3D images.
# Here, we will train on 2D slices from the 3D images, so we use the SliceSampler
# We will serve batches of 32 slices from our images and the slices will be shuffled.
# As we will see later, the loader will read in 4 images at a time, take all the slices from those
# 4 images and shuffle them together, then serve them in batches of 32 until the model has
# consumed all the slices from those 4 images. Then, the next 4 images from th dataset will be
# read in and the slice batching process will run again. That will happen until all images from
# the dataset have been read in. 
sampler = samplers.SliceSampler(axis=0, sub_batch_size=32, shuffle=True)

### create loader
# We create the dataset loader which acts as the actual data generator for the model.
# The loader will serve numpy arrays with the final dimension expanded. As mentioned above,
# because we are using a sampler then the batch_size for the loader does not determine the
# array size that the loader will serve, but instead determines how many images are read in
# together to then take slices from. 
# Notice that we also have transforms here, similar to those when we created a dataset. The
# purpose of transforms in the loader is to augment the dataset by applying RANDOM transforms
# to the images each time they're served. These are transforms that are meant to be applied
# only during training, since they can greatly distort the images in order to make the model
# more robust to different images.
loader = loaders.DatasetLoader(dataset=dataset,
                               batch_size=4,
                               sampler=sampler,
                               expand_dims=-1,
                               x_transforms=[tx.RandomNoise(0, 2),
                                             tx.RandomFlip(p=0.5),
                                             tx.RandomSmooth(0, 2)])

# create model manually

# fit model manually 
