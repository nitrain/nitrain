# Nitrain - neuroimaging tools for deep learning

Nitrain provides tools for sampling and augmenting neuroimaging datasets, training deep learning models on neuroimages, and visualizing or explaining deep learning model results in a neuroimaging context. Nitrain also makes it easy to use pre-trained models for training or integration into imaging pipelines.

## Installation

The package can be installed from github:

```
python -m pip install git+github.com/ncullen93/nitrain.git
```

## Quickstart

Let's say you want to predict a disease-related phenotype from structural T1 images using a convolutional neural network in Pytorch. You have a cohort of 200 participants and to avoid overfitting you want to perform neuroimage-specific data augmentation (e.g., affine transforms) during training. When the model is finalized, you'd like to understand which parts of the brain the model is paying attention to most.

This is a canonical example for using nitrain. Here's how it would look like:

```python
from nitrain import sample, train, explain

```

## Sampling and augmentation

Nitrain provides extensive functionality to help you sample neuroimages with imaging-native data augmentation techniques. Our focus is on speed, meaning you never have to convert your neuroimages into numpy arrays. Here is an example of how sampling works:

```python
import nitrain

```

### Types of transforms

The nitrain package supports an extensive amount of neuroimaging-based transforms:

- Affine (Rotate, Translate, Shear, Zoom)
- Flip, Pad, Crop, Slice
- Noise
- Motion
- Intensity normalization

But nitrain also supports more extreme transformations such as those derived from non-linear image registration using the [https://github.com/antsx/ants][ANTs] framework.

All of these transforms can be combined using a single `Compose` transform that ensures efficient augmentation during model training.

## Training with popular frameworks

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

## Starting from pre-trained models

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

## Visualizing model results

The idea that deep learning models are "black boxes" is out-dated, particularly when it comes to images. There are numerous techiques to help you understand which parts of the brain a trained model is weighing most when making predictions.

One such technique is called the occlusion method, where you systematically "black out" different patches of an input image and see how the model prediction is affected. The idea is that when, when occluded, important areas result in a large change in model prediction compared to the original image.

Nilearn provides tools to perform this techique - along with many others - and can help you visualize the results of such explainability experiments directly in brain space. Here is what that might look like:

```python
from nitrain import explain
```

## Contributing

If you would like to contribute to nilearn, we would be extremely thankful. The best way to start is by posting an issue to discuss your proposed feature. We use Poetry as our build tool.
