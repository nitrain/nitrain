from .base_transform import BaseTransform

class ToFile(BaseTransform):

    def __init__(self, dir):
        """
        Saves an image to file. Useful as a pass-through ransform
        when wanting to observe how augmentation affects the data
        """
        self.dir = dir

    def __call__(self, *inputs):
        pass


class CustomFunction(BaseTransform):
    
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, image):
        new_image = self.fn(image)
        return new_image


class RandomChoice(BaseTransform):
    
    def __init__(self, *transforms, prob=None):
        self.transforms = transforms
        if prob is None:
            self.prob = 1.0 / len(self.transforms)
    
    def __call__(self, image):
        pass