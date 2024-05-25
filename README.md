# Nitrain: a medical imaging-native AI framework

[![Coverage Status](https://coveralls.io/repos/github/nitrain/nitrain/badge.svg?branch=main)](https://coveralls.io/github/nitrain/nitrain?branch=main)
[![Build](https://github.com/nitrain/nitrain/actions/workflows/test.yml/badge.svg)](https://github.com/nitrain/nitrain/actions/workflows/test.yml)

Nitrain (formerly <i>torchsample</i>) is a framework-agnostic python library for sampling and augmenting medical images, training models on medical imaging datasets, and visualizing results in a medical imaging context.

The nitrain library is unique in that it makes training models as simple as possible by providing reasonable defaults and a high-level of abstraction. It also supports multiple frameworks - torch, tensorflow, and keras - with a goal to add even more.

Full examples of training medical imaging AI models using nitrain can be found at the [tutorials](https://github.com/nitrain/tutorials) repository. If you are interested more generally in medical imaging AI, check out the book [Becoming a medical imaging AI expert with Python](https://book.nitrain.dev).

<br />

## Quickstart

Here is a canonical example of using nitrain to a semantic segmentation model. Notice how easy it is to map image files from a local folder and how straight-forward it is to sample batches of augmented, 2D slices from 3D images.

```python
import nitrain as nt
from nitrain.readers import ImageReader, ColumnReader

# create dataset from folder of images + participants file
dataset = nt.Dataset(inputs=ImageReader('sub-*/anat/*_T1w.nii.gz'),
                     outputs=ImageReader('sub-*/anat/*_aparc+aseg.nii.gz'),
                     transforms={
                         'inputs': tx.NormalizeIntensity(0,1),
                         ('inputs', 'outputs'): tx.Resize((64,64,64))
                     },
                     base_dir='~/desktop/ds004711/')

# create loader with random transforms
loader = nt.Loader(dataset,
                   images_per_batch=4,
                   sampler=nt.SliceSampler(batch_size = 32, axis = 2)
                   transforms={
                           'inputs': tx.RandomNoise(sd=0.2)
                   })

# create model from architecture
arch_fn = nt.fetch_architecture('unet', dim=2)
model = arch_fn(input_image_size=(64,64,1),
                mode='segmentation')

# create trainer and fit model
trainer = nt.Trainer(model, task='segmentation')
trainer.fit(loader, epochs=100)

# upload trained model to platform
nt.register_model(trainer.model, 'nick/t1-brain-segmentation')
```

If you want to learn a bit more about key components of nitrain then you can follow the 10-minute overview tutorial further below. Also, a large variety of self-contained notebooks showing how to perform common medical imaging AI tasks is available in the [tutorials](https://www.github.com/nitrain/tutorials) repo.

<br />

## Installation

The latest release of nitrain can be installed from pypi:

```
pip install nitrain
```

Or you can install the latest development version directly from github:

```
python -m pip install git+github.com/nitrain/nitrain.git
```

### Dependencies

The [ants](https://www.github.com/antsx/antspy) python package is a key dependency that allows you to efficiently read, operate on, and visualize medical images. Additionally, you can use keras (tf.keras or keras3), tensorflow, or pytorch as backend for creating your models.

<br />

## Resources

The following links can be helpful in becoming more familiar with nitrain.

- Introduction tutorials [[Link](https://github.com/nitrain/tutorials/tree/main/introduction)]
- Classification examples [[Link](https://github.com/nitrain/tutorials/tree/main/classification)]

<br />

## Contributing

If you have a question, feature request, or bug report the best way to get help is by posting an issue on the GitHub page. We welcome any new contributions and ideas to nitrain. If you want to add code, the best way to get started is by posting an issue or contacting me at nickcullen31@gmail.com.
