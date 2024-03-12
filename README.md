# Nitrain - neuroimaging tools for deep learning

Nitrain provides tools for sampling and augmenting neuroimaging datasets, training deep learning models on neuroimages, and visualizing or explaining deep learning model results in a neuroimaging context. Nitrain also makes it easy to use pre-trained models for training or integration into imaging pipelines.

<br />

## Installation

The package can be installed from github:

```
python -m pip install git+github.com/ncullen93/nitrain.git
```

### Dependencies

The nitrain package is developed with Python3.10 as the focus. In terms of dependencies, it requires `ants` and `antspynet` and works with either tensorflow/keras or pytorch. Support for Keras 3 is on the roadmap.

<br />

## Quickstart

Let's say you want to predict a disease-related phenotype from structural T1 images using a convolutional neural network in Pytorch. You have a cohort of 200 participants and to avoid overfitting you want to perform neuroimage-specific data augmentation (e.g., affine transforms) during training. When the model is finalized, you'd like to understand which parts of the brain the model is paying attention to most.

This is a canonical example for using nitrain. Here's how it would look like:

```python
from nitrain import datasets, loaders, models, trainers, transforms as tx

# create dataset from folder of images + participants file
dataset = datasets.FolderDataset(base_dir='ds004711',
                            x={'pattern': 'sub-*/anat/*_T1w.nii.gz'},
                            y={'file': 'participants.tsv',
                                    'column': 'age'},
                            x_transforms=[nit.ResizeImage((64,64,64)),
                                          nit.NormalizeIntensity(0,1)])

# create loader with random transforms
loader = loaders.DatasetLoader(dataset,
                           batch_size=32,
                           shuffle=True,
                           x_transforms=[nit.RandomSlice(axis=2),
                                         nit.RandomNoise(sd=0.2)])

# create model from architecture
architecture_fn = models.fetch_architecture('alexnet', task='continuous_prediction')
model = architecture_fn(layers=[128, 64, 32, 10], n_outcomes=1)

# create trainer and fit model
trainer = trainers.ModelTrainer(model,
                           loss='mse',
                           optimizer='adam',
                           lr=1e-3,
                           callbacks=[nit.EarlyStopping(),
                                      nit.ModelCheckpoints(freq=25)])
trainer.fit(loader, epochs=100)

# upload trained model to platform
models.register_model(trainer.model, 'nick/t1-brainage-model')
```

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
                               expand_dim=-1,
                               x_transforms=[tx.RandomSmoothing(0, 1)])
```

The loader can be be used directly as a batch generator to fit models in tensorflow, keras, pytorch, or any other framework. Note that we also have loaders geared specifically towards those frameworks to allow you to use some additional loading functionality that they provide.

### Types of transforms

The philosophy of nitrain is to be as neuroimaging-native as possible. That means that all transforms are applied directly on images (specifically, `antsImage` types from the [https://github.com/antsx/antspy][ANTsPy] package) and only at the very end of batch generator are the images converted to numpy arrays.

The nitrain package supports an extensive amount of neuroimaging-based transforms:

- Affine (Rotate, Translate, Shear, Zoom)
- Flip, Pad, Crop, Slice
- Noise
- Motion
- Intensity normalization

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

## Trainers

Nitrain can be used to train models on Pytorch, Keras, and Tensorflow.

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

## Architectures and Pre-trained models

We provide a collection of pre-trained models that may prove extremely useful as a starting point for your training. If your dataset is small (<500 participants) than you may especially benefit from using our pre-trained models, since they have already learned the basic patterns of a neuroimage. Fine-tuning a pre-trained model is simple:

```python
import nitrain
model = nitrain.fetch_pretrained('basic-t1')
```

If you have trained an interested deep learning model on neuroimages and would like to share it with the community, it is possible to do so directly from nitrain. Any model you share will be hosted by us and available for use by anyone else through the `fetch_pretrained` function.

```python
import nitrain
nitrain.register_pretrained(model, 'my-cool-model')
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

If you would like to contribute to nitrain, we would be extremely thankful. The best way to start is by posting an issue to discuss your proposed feature. We use Poetry as our build tool.
