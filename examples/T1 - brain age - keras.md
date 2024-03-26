# Predicting age from T1w brain images with nitrain and keras

This is a fully self-contained example that demonstrates how to fit a model to predict chronological age from a T1-weighted structural brain image. This a classic image-to-continuous regression problem that demonstrates much of the core functionality that nitrain provides.

Let's start by importing the packages we will need. The model will be built and training using Keras v2 via tensorflow.


```python
import os
import math
from nitrain import datasets, loaders, models, samplers, transforms as tx
import tensorflow as tf
```

The dataset used in this example is `ds004711` from the open-source OpenNeuro repository. We provide the ability to download any OpenNeuro dataset locally using Datalad. By using datalad, only symbolic links to the images will be downloaded instead of the entire images themselves. 

This means the data download will be very quick and wont take up much space -- nitrain will take care of downloading the images needed the first time they're used during batch generation.


```python
work_dir = os.path.expanduser('~/Desktop')
datasets.fetch_data('openneuro/ds004711', path=work_dir)
```

Next, we create a `BIDSDataset` from the downloaded dataset. This is similar to a `FolderDataset` but uses the `pybids` package to provide some more convenient ways to identify the data you want to use. 

Here, the input (x) to the model will be the first T1 image for each subject (n = 187). As mentioned, the input files are selected using convenient bids entities (see pybids package), but if you didnt have a BIDS
dataset then you can instead grab files uing glob patterns with a FolderDataset.

The output (y) will be the age values for the subjects taken from the participants.tsv file.

As we see with the `x_transforms` argument, when the T1 images (x) are first read from file, they will be downsampled by 4x and skull stripped. The x_transforms passed to a dataset should be non-random, preprocessing steps that ideally would be done beforehand. And indeed it is possible to precompute the dataset transforms (not shown here).


```python
dataset = datasets.BIDSDataset(os.path.join(work_dir, 'openneuro/ds004711'),
                               x={'suffix': 'T1w', 'run': [None, '01']},
                               x_transforms=[tx.Resample((4,4,4), use_spacing=True),
                                             tx.BrainExtraction()],
                               y={'file': 'participants.tsv', 'column': 'age'},
                               datalad=True)
```

If you want to see what a dataset does, you can access it via standard python indexing. This makes it clear what a dataset does: reads an image from file (or whatever the source may be) then applies the transforms in order one at a time.


```python
x_raw, y_raw = dataset[:3]
```

The next thing to do is to pass a dataset into a loader. But since we want to try our model on 2D slices of the original 3D images, we need a sampler. Samplers are needed if you want to train your model on something other than the full image.

This means you need samplers if you want to train on 1) 2D slices of 3D images, 2) 3D blocks
3D images, 3) 2D patches of 2D images, or 4) 2D patches of 3D images.

We will use a `SliceSampler` to serve shuffled batches of 32 slices from our images. As we will see later, the loader will read in 4 images at a time, take all the slices from those 4 images and shuffle them together, then serve them in batches of 32 until the model has consumed all the slices from those 4 images. Then, the next 4 images from the dataset will be read in and the slice batching process will run again. That will happen until all images from the dataset have been read in. 


```python
sampler = samplers.SliceSampler(axis=2, sub_batch_size=32, shuffle=True)
```

Now we create the dataset loader which acts as the actual data generator for the model. The loader will serve numpy arrays with the final dimension expanded. As mentioned above, because we are using a sampler then the batch_size for the loader does not determine the array size that the loader will serve, but instead determines how many images are read in together to then take slices from. 

Notice that we also have transforms here, similar to those when we created a dataset. The purpose of transforms in the loader is to augment the dataset by applying RANDOM transforms to the images each time they're served. These are transforms that are meant to be applied only during training, since they can greatly distort the images in order to make the model more robust to different images.


```python
loader = loaders.DatasetLoader(dataset=dataset,
                               batch_size=4,
                               sampler=sampler,
                               x_transforms=[tx.RandomNoise(0, 2),
                                             tx.RandomFlip(p=0.5),
                                             tx.RandomSmoothing(0, 2)])
```

Just as we did with the dataset, we can read in the first batch to see what it looks like.

The `x_batch` variable is a numpy array with shape (32, 64, 64, 1) - it is 32 random slices from the first 4 images with the last dimension expanded for model training purposes (optional: see `expand_dims` argument).

The `y_batch` is a numpy array of length 32 - it is the age each slice's source participant.

Note that there are 187 participants, 48 slices per participant, and a batch size of 32. Therefore, one training epoch will have 187 * 48 / 32 = 280.5 => 281 batches and will see 8976 slices in total. Luckily, you don't need to worry about this math by setting a `steps_per_epoch` argument in your model fit function or anything like that... the loader takes care of that.


```python
x_batch, y_batch = next(iter(loader))
```

The model here is created based on the Alexnet architecture and is made in tf.Keras. As you can see, the architectures are a sort of blueprint that let you flexibly create a model of a given type based on your task (classification, regression, etc) and the size and shape of your data.

There are many different architectures that you can choose from -- just run `models.list_architectures()` to explore what we have. Of course, if you want to create your own model from scratch then you are free to do so!


```python
arch_fn = models.fetch_architecture('vgg', dim=2)
model = arch_fn(input_image_size=(64,64,1), 
                number_of_classification_labels=1,
                mode='regression')
```

And finally, we can compile and fit the model. Since we have a Keras model, we convert our loader to a KerasLoader type. We could've also started with the `KerasLoader` class instead of the `DatasetLoader` class, but this way shows a bit more the flexibility of nitrain.

The model fitting procedure is done here using the standard keras workflow. However, nitrain also provides the `trainers.ModelTrainer` class which acts as a unified, high-level interface for training pytorch, keras, or tensorflow models in the same way.


```python
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

model.fit(loader.to_keras(), epochs=100)
```

Hopefully this demonstrates just how easy it easy to augment and serve batches of medical imaging data using nitrain. This example also quickly showed how you can create models from well-known architectures.
