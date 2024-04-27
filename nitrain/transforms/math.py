
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
        images = [image.ceil() for image in images]
        return images if len(images) > 1 else images[0]


class Floor(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, *images):
        images = [image.floor() for image in images]
        return images if len(images) > 1 else images[0]


class Log(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, *images):
        images = [image.log() for image in images]
        return images if len(images) > 1 else images[0]


class Exp(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, *images):
        images = [image.exp() for image in images]
        return images if len(images) > 1 else images[0]


class Sqrt(BaseTransform):
    def __init__(self):
        pass
    def __call__(self, *images):
        images = [image.sqrt() for image in images]
        return images if len(images) > 1 else images[0]


class Power(BaseTransform):
    def __init__(self, value):
        self.value = value
    def __call__(self, *images):
        images = [image.power(self.value) for image in images]
        return images if len(images) > 1 else images[0]