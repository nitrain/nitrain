# Nitrain - a framework for medical imaging-native AI

[![Coverage Status](https://coveralls.io/repos/github/ncullen93/nitrain/badge.svg?branch=main)](https://coveralls.io/github/ncullen93/nitrain?branch=main)
[![Build](https://github.com/ncullen93/nitrain/actions/workflows/test.yml/badge.svg)](https://github.com/ncullen93/nitrain/actions/workflows/test.yml)
![PyPI - Downloads](https://img.shields.io/pypi/dm/nitrain)

Nitrain is a medical imaging-native framework for deep learning. It supports pytorch, keras, and tensorflow and complements them by providing tools for sampling and augmenting medical images, training models, and visualizing model results in a medical imaging context.

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

The nitrain package uses the ANTs python package to efficiently read and transform medical images. Additionally, we support keras (tf.keras or keras3), tensorflow, and pytorch for all core functionality.

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

## Datasets and Loaders

Nitrain provides extensive functionality to help you sample neuroimages with imaging-native data augmentation techniques. Our focus is on speed, meaning you never have to convert your neuroimages into numpy arrays.

You start by creating a `dataset` from wherever your images are stored -- in a local folder, in a bids folder, in memory, on a cloud service, etc. Say that your data is stored in a local folder. To grab inputs (`x`), supply a dictionary with a glob pattern and optional exclude pattern. To grab outputs (`y`), specify a dataframe file to read and a column to pull from that dataframe. Both x and y can be image configs, and they can also be lists of configs if you want to grab multiple images at once.

```python
from nitrain import datasets
dataset = datasets.FolderDataset(base_dir='ds004711',
                                 x={'pattern': 'sub-*/anat/*_T1w.nii.gz', 'exclude': '**run-02*'},
                                 y={'file': 'participants.tsv', 'column': 'age'},
                                 x_transforms=[tx.Resample((4,4,4), use_spacing=True),
                                               tx.RangeNormalize(0,1)])

```

Notice also that there the `x_transforms` argument has been supplied. These transforms are applied every time the input image is read into file. These are meant to be "fixed" transforms - i.e., not random - because the results can be cached to speed up sampling in the long-run.

Once you have a dataset, you can grab images from it as you would with any iterator. This gives you the first input image (with transforms applied) and the first age value.

```python
x, y = dataset[0]
```

The dataset can then be passed into a `loader` in order to actually sample batches. With loaders, you can specify parameters like batch size and whether to expand dims. You can also pass in more transforms that will be applied at each batch sampling. These transforms, in contrast, are meant to be random data augmentation transforms.

```python
loader = loaders.DatasetLoader(dataset,
                               batch_size=32,
                               x_transforms=[tx.RandomSmoothing(0, 1)],
                               expand_dims=-1)

# loop through all images in batches for one epoch
for x_batch, y_batch in loader:
        print(y_batch)
```

The loader can be be used directly as a batch generator to fit models in tensorflow, keras, pytorch, or any other framework. Note that we also have loaders geared specifically towards those frameworks to allow you to use some additional loading functionality that they provide.

<br />

## Samplers

Samplers allow you to keep the same dataset + loader workflow that batches entire images and applies transforms to them, but then expand on those transformed image batches to create special "sub-batches".

For instance, samplers let you serve batches of 2D slices from 3D images, or 3D blocks from 3D images, and so forth. Samplers are essntial for common deep learning workflows in medical imaging where you often want to train a model on only parts of the image at once.

All you have to do is supply a sampler instance to your dataset loader. Here is an example:

```python
from nitrain import loaders, samplers as sp, transforms as tx
loader = loaders.DatasetLoader(dataset,
                               batch_size=3,
                               x_transforms=[tx.RandomSmoothing(0, 1)],
                               expand_dims=-1,
                               sampler=sp.SliceSampler(sub_batch_size=24, axis=0, shuffle=True))
```

What happens is that we start with the ~190 images from the dataset, but 3 images will be read in from file at a time. Then, all possible 2D slices will be created from those 3 images and served in shuffled batches of 24 from the loader. Once all "sub-batches" (sets of 24 slices from the 3 images) have been served, the loader will move on to the next 3 images and serve slices from those images. One epoch is completed when all slices from all images have been served.

The important thing to remember is that the batch size your model will see is 24. In total, then, there are (n_images \* n_slices_per_image / sampler_batch_size) total batches in one epoch instead of (n_images / loader_batch_size) like there normally are.

<br />

## Transforms

The philosophy of nitrain is to be as neuroimaging-native as possible. That means that all transforms are applied directly on images - specifically, `antsImage` types from the [ANTsPy](https://github.com/antsx/antspy) package - and only at the very end of batch generator are the images converted to numpy arrays.

The nitrain package supports an extensive amount of neuroimaging-based transforms:

- Affine (Rotate, Translate, Shear, Zoom)
- Flip, Pad, Crop, Slice
- Noise
- Motion
- Intensity normalization

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

Writing your own transform is extremely easy! Just remember that the transform will operate on the `antsImage` type and that you should inherit from the `BaseTransform` class.

```python
from nitrain.transforms import BaseTransform

class CoolTransform(BaseTransform):
        def __init__(self, parameters):
                self.parameters = parameters
        def __call__(self, image):
                image = my_function(image, self.parameters)
                return image

tx_fn = CoolTransform(parameters=123)
img_transformed = tx_fn(img)
```

<br />

## Architectures and pretrained models

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

There is also a large collection of pretrained models available as a starting point for your training or simply to use for inference. If your dataset is small (<500 participants) than you may especially benefit from using pre-trained models.

Similarly to architectures, you fetch a pretrained model based on its name. The result of fetching a pretrained model is the actual instantianed model with the pretrained weights loaded.

```python
from nitrain import models
model = models.fetch_pretrained('basic-t1')
```

If you have trained an interested deep learning model on neuroimages and would like to share it with the community, it is possible to do so directly from nitrain. Any model you share will be hosted and available for use by anyone else through the `fetch_pretrained` function.

```python
from nitrain import models
models.register_pretrained(model, 'my-cool-model')
```

<br />

## Model trainers

After you have either fetched and created an architecture, fetched a pretrained model, or created a model yourself in your framework of choice, then it's time to actually train the model on the dataset / loader that you've created.

To train with Pytorch, use the `nitrain.torch` module:

```python
import nitrain
```

To train with Keras, use the `nitrain.keras` module:

```python
import nitrain
```

To train with Tensorflow, use the `nitrain.tensorflow` module:

```python
import nitrain
```

<br />

## Explainers

The idea that deep learning models are "black boxes" is out-dated, particularly when it comes to images. There are numerous techiques to help you understand which parts of the brain a trained model is weighing most when making predictions.

One such technique is called the occlusion method, where you systematically "black out" different patches of an input image and see how the model prediction is affected. The idea is that when, when occluded, important areas result in a large change in model prediction compared to the original image.

Nitrain provides tools to perform this techique - along with many others - and can help you visualize the results of such explainability experiments directly in brain space. Here is what that might look like:

```python
from nitrain import explain
```

<br />

## Contributing

If you would like to contribute to nitrain, we would be extremely thankful. The best way to start is by posting an issue to discuss your proposed feature.
