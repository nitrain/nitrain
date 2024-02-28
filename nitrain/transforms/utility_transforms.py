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