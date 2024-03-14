# This is a complete, self-contained example of training a brain-age model
# using the full data loading functionality from nitrain.
# This example demonstrates nitrain's `datasets`, `loaders`, `samplers`, and `transforms`,
# but it lets you use your own code to create and train the model.

import os
import math
from nitrain import datasets, loaders, models, samplers, transforms as tx
import tensorflow as tf

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
                               x_transforms=[tx.Resample((4,4,4), use_spacing=True),
                                             tx.BrainExtraction()],
                               y={'file': 'participants.tsv', 'column': 'age'},
                               y_transforms=[tx.CustomFunction(lambda age: [int(age > 50), int(age < 50)])],
                               datalad=True)

# read in and transform the first three images + ages to see what it looks like
# x_raw is a list of resampled + brain extract images; y_raw is a np array of age classifications
x_raw, y_raw = dataset[:3]
dataset.x = dataset.x[:100]
dataset.y = dataset.y[:100]

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
sampler = samplers.SliceSampler(axis=2, sub_batch_size=32, shuffle=True)

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
                               x_transforms=[tx.RandomNoise(0, 2),
                                             tx.RandomFlip(p=0.5),
                                             tx.RandomSmoothing(0, 2)])

# read in the first batch to see what it looks like
# x_batch is a np array with shape (32, 64, 64, 1) - it is 32 random slices from the first 4 images
# with the last dimension expanded for model training purposes (optional: see `expand_dims` argument)
# y_batch is a np array of length 32 - it is the age classification each slice's source participant
# Note that there are 187 participants, 48 slices per participant, and a batch size of 32. Therefore,
# one training epoch will have 187 * 48 / 32 = 280.5 => 281 batches and will see 8976 slices in total.
x_batch, y_batch = next(iter(loader))

### create model 
# The model here is created based on the provided Alexnet architecture. As you can see, the
# architectures are a sort of blueprint that let you flexibly create a model of a given type
# based on your task (classification, regression, etc) and the size and shape of your data.
arch_fn = models.fetch_architecture('alexnet', dim=2)
model = arch_fn(input_image_size=(64,64,1), 
                number_of_classification_labels=2,
                mode='classification')

# compile and fit model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(loader.as_keras_loader(), steps_per_epoch=150, epochs=10)
