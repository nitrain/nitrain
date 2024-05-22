
import ants
import numpy as np

from .base import BaseTransform

__all__ = [
    'Abs',
    'Ceil',
    'Floor',
    'Log',
    'Exp',
    'Sqrt',
    'Power'
]


class Abs(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, *images):
        images = [image.abs() for image in images]
        return images if len(images) > 1 else images[0]


class Ceil(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, *images):
        images = [ants.from_numpy_like(np.ceil(image.numpy()), image) for image in images]
        return images if len(images) > 1 else images[0]


class Floor(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, *images):
        images = [ants.from_numpy_like(np.floor(image.numpy()), image) for image in images]
        return images if len(images) > 1 else images[0]


class Log(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, *images):
        images = [ants.from_numpy_like(np.log(image.numpy()), image) for image in images]
        return images if len(images) > 1 else images[0]


class Exp(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, *images):
        images = [ants.from_numpy_like(np.exp(image.numpy()), image) for image in images]
        return images if len(images) > 1 else images[0]


class Sqrt(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, *images):
        images = [ants.from_numpy_like(np.sqrt(image.numpy()), image) for image in images]
        return images if len(images) > 1 else images[0]


class Power(BaseTransform):
    def __init__(self, value):
        self.value = value
    def __call__(self, *images):
        images = [ants.from_numpy_like(np.power(image.numpy(), self.value), image) for image in images]
        return images if len(images) > 1 else images[0]