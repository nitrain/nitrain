# Nitrain - a framework for medical imaging-native AI

[![Coverage Status](https://coveralls.io/repos/github/ncullen93/nitrain/badge.svg?branch=main)](https://coveralls.io/github/ncullen93/nitrain?branch=main)
[![Build](https://github.com/ncullen93/nitrain/actions/workflows/test.yml/badge.svg)](https://github.com/ncullen93/nitrain/actions/workflows/test.yml)
![PyPI - Downloads](https://img.shields.io/pypi/dm/nitrain)

Nitrain (formerly <i>torchsample</i>) provides tools for sampling and augmenting medical images, training models on medical imaging datasets, and visualizing model results in a medical imaging context. It supports using pytorch, keras, and tensorflow.

You can also train models HIPAA-compliantly in the cloud using nitrain. You are encouraged to make your fitted models available to the community via nitrain, or you can easily use other's pretrained models for fine-tuning or standard image processing.

<br />

## Installation

The latest release of nitrain can be installed from pypi:

```
pip install nitrain
```

Or you can install the latest development version directly from github:

```
python -m pip install git+github.com/ncullen93/nitrain.git
```

### Dependencies

The nitrain package uses the `antspy` package to efficiently read and transform medical images. It relies on the `antspynet` package to create some architectures. Additionally, you can use keras (tf.keras or keras3), tensorflow, or pytorch as backend - with support for only importing the framework you are using.

<br />

## Quickstart

Here is a canonical example of using nitrain to fit a brain-age model. If you want to learn a bit more about key components of nitrain then you can follow the overview tutorials just below the quickstart.

```python
from nitrain import datasets, loaders, models, trainers, transforms as tx

# create dataset from folder of images + participants file
dataset = datasets.FolderDataset(base_dir='ds004711',
                                 x={'pattern': 'sub-*/anat/*_T1w.nii.gz'},
                                 y={'file': 'participants.tsv',
                                         'column': 'age'},
                                 x_transforms=[tx.Resize((64,64,64)),
                                               tx.NormalizeIntensity(0,1)])

# create loader with random transforms
loader = loaders.DatasetLoader(dataset,
                               batch_size=4,
                               shuffle=True,
                               sampler=samplers.SliceSampler(sub_batch_size=32, axis=2)
                               x_transforms=[tx.RandomNoise(sd=0.2)])

# create model from architecture
arch_fn = models.fetch_architecture('alexnet', dim=2)
model = arch_fn(input_image_size=(64,64,1),
                number_of_outcomes=1,
                mode='regression')

# create trainer and fit model
trainer = trainers.ModelTrainer(model,
                                loss='mse',
                                optimizer='adam',
                                lr=1e-3,
                                callbacks=[utils.EarlyStopping(),
                                           utils.ModelCheckpoints(freq=25)])
trainer.fit(loader, epochs=100)

# upload trained model to platform
models.register_model(trainer.model, 'nick/t1-brain-age')
```

A more in-depth introduction can be found in the [tutorials](github.com/ncullen93/nitrain) and if you can also check out the [examples](github.com/ncullen93/nitrain) for self-contained notebooks showing how to perform common deep learning tasks.

<br />

## Overview of nitrain

The 10-minute overview presented below will take you through the key components of nitrain:

- [Datasets and Loaders](#datasets-and-loaders)
- [Samplers](#samplers)
- [Transforms](#transforms)
- [Architectures and pretrained models](#architectures-and-pretrained-models)
- [Model trainers](#model-trainers)
- [Explainers](#explainers)

<br />

### Datasets and Loaders

Datasets help you read in your images from wherever they are stored -- in a local folder with BIDS or datalad, in memory, on a cloud service. You can flexibly specify the inputs and outputs using glob patterns, BIDS entities, etc. Transforms can also be passed to your datasets as a sort of preprocessing pipeline that will be applied whenever the dataset is accessed.

```python
from nitrain import datasets, transforms as tx

dataset = datasets.FolderDataset(base_dir='~/datasets/ds004711',
                                 x={'pattern': 'sub-*/anat/*_T1w.nii.gz', 'exclude': '**run-02*'},
                                 y={'file': 'participants.tsv', 'column': 'age'},
                                 x_transforms=[tx.Resample((64,64,64))])
```

Although you will rarely need to do this, data can be read into memory by indexing the dataset:

```python
x_raw, y_raw = dataset[:3]
```

To prepare your images for batch generation during training, you pass the dataset into one the loaders. Here is where you can also pass in random transforms that will act as data augmentation. If you want to train on slices, patches, or blocks of images then you will additionally provide a sampler. The different samplers are explained later.

```python
from nitrain import loaders, samplers

loader = loaders.DatasetLoader(dataset,
                               batch_size=32,
                               x_transforms=[tx.RandomSmoothing(0, 1)])

# loop through all images in batches for one epoch
for x_batch, y_batch in loader:
        print(y_batch)
```

The loader can be be used directly as a batch generator to fit models in tensorflow, keras, pytorch, or any other framework.

<br />

### Samplers

Samplers allow you to keep the same dataset + loader workflow that batches entire images and applies transforms to them, but then expand on those transformed image batches to create special "sub-batches".

For instance, samplers let you serve batches of 2D slices from 3D images, or 3D blocks from 3D images, and so forth. Samplers are essntial for common deep learning workflows in medical imaging where you often want to train a model on only parts of the image at once.

```python
from nitrain import loaders, samplers, transforms as tx
loader = loaders.DatasetLoader(dataset,
                               batch_size=4,
                               x_transforms=[tx.RandomSmoothing(0, 1)],
                               sampler=samplers.SliceSampler(sub_batch_size=24, axis=2))
```

What happens is that we start with the ~190 images from the dataset, but 4 images will be read in from file at a time. Then, all possible 2D slices will be created from those 4 images and served in shuffled batches of 24 from the loader. Once all "sub-batches" (sets of 24 slices from the 4 images) have been served, the loader will move on to the next 4 images and serve slices from those images. One epoch is completed when all slices from all images have been served.

<br />

### Transforms

The philosophy of nitrain is to be medical imaging-native. This means that all transforms are applied directly on images - specifically, `antsImage` types from the [ANTsPy](https://github.com/antsx/antspy) package - and only at the very end of batch generator are the images converted to arrays / tensors for model consumption.

The nitrain package supports an extensive amount of medical imaging transforms:

- Affine (Rotate, Translate, Shear, Zoom)
- Flip, Pad, Crop, Slice
- Noise
- Motion
- Intensity normalization

You can create your own transform with the `CustomTransform` class:

```python
from nitrain import transforms as tx

my_tx = tx.CustomTransform(lambda x: x * 2)
```

If you want to explore what a transform does, you can take a sample of it over any number of trials on the same image and then plot the results:

```python
import ants
import numpy as np
from nitrain import transforms as tx

img = ants.image_read(ants.get_data('r16'))

my_tx = tx.RandomSmoothing(0, 2)
imgs = my_tx.sample(img, n=12)

ants.plot_grid(np.array(imgs).reshape(4,3))
```

<br />

### Architectures and pretrained models

The nitrain package provides an interface to an extensive amount of deep learning model architectures for all kinds of tasks - regression, classification, image-to-image generation, segmentation, autoencoders, etc.

The available architectures can be listed and explored:

```python
from nitrain import models
print(models.list_architectures())
```

You first fetch an architecture function which provides a blueprint on creating a model of the given architecture type. Then, you call the fetched architecture function in order to actually create a specific model with you given parameters.

```python
from nitrain import models

vgg_fn = models.fetch_architecture('vgg', dim=3)
vgg_model = vgg_fn((128, 128, 128, 1))

autoencoder_fn = models.fetch_architecture('autoencoder')
autoencoder_model = autoencoder_fn((784, 500, 500, 2000, 10))
```

<br />

### Trainers

After you have created a model from a nitrain architecture, fetched a pretrained model, or created a model yourself in your framework of choice, then it's time to actually train the model on the dataset / loader that you've created.

Although you are free to train models on loaders using standard pytorch, keras, or tensorflow workflows, we also provide the `ModelTrainer` class to make training even easier. This class provides sensible defaults for key training parameters based on your task.

```python
trainer = trainers.ModelTrainer(model=vgg_model, task='regression')
trainer.fit(loader, epochs=10)

# access fitted model
print(trainer.model)
```

Additionally, you can train your model in the cloud using the `CloudTrainer` class. All training takes place on HIPAA-compliant servers.

<br />

### Explainers

The idea that deep learning models are "black boxes" is out-dated, particularly when it comes to images. There are numerous techiques to help you understand which parts of the brain a trained model is weighing most when making predictions.

Nitrain provides tools to perform this techique - along with many others - and can help you visualize the results of such explainability experiments directly in brain space. Here is what that might look like:

<br />

## Contributing

If you would like to contribute to nitrain, we would be extremely thankful. The best way to start is by posting an issue to discuss your proposed feature.
