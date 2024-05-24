import ants

from .base import BaseTransform

__all__ = [
    'AddChannel',
    'Reorient'
]

class AddChannel(BaseTransform):
    def __init__(self, axis=-1):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.AddChannel()
        img2 = mytx(img)
        """
        self.axis = axis
        
    def __call__(self, *images):
        images = [ants.merge_channels([image]) for image in images]
        return images if len(images) > 1 else images[0]

class Reorient(BaseTransform):
    def __init__(self, orientation):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Reorient('RAS')
        img2 = mytx(img)
        """
        self.orientation = orientation
        
    def __call__(self, *images):
        images = [image.reorient_image2(self.orientation) for image in images]
        return images if len(images) > 1 else images[0]