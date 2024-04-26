
from .base import BaseTransform

__all__ = [
    'Astype',
    'Smooth',
    'Crop',
    'Resample',
    'Slice'
]


class Astype(BaseTransform):
    """
    import ntimage as nti
    import nitrain as nt
    from nitrain import transforms as tx
    from nitrain.readers import ImageReader
    dataset = nt.Dataset(
        inputs={'x':ImageReader([nti.example('mni') for _ in range(10)]),
                'y':ImageReader([nti.example('mni') for _ in range(10)])},
        outputs=ImageReader([nti.example('mni') for _ in range(10)]),
        transforms={
            'x': tx.Astype('uint8')
        }
    )
    x, y = dataset[0]
    dataset2 = nt.Dataset(
        inputs=[ImageReader([nti.example('mni') for _ in range(10)]),
                ImageReader([nti.example('mni') for _ in range(10)])],
        outputs=ImageReader([nti.example('mni') for _ in range(10)]),
        transforms={
            ('inputs','outputs'): [tx.Astype('uint8')]
        }
    )
    x, y = dataset[0]
    """
    def __init__(self, dtype):
        self.dtype = dtype
    def __call__(self, *images):
        images = [image.astype(self.dtype) for image in images]
        return images if len(images) > 1 else images[0]

class Smooth(BaseTransform):
    pass

class Crop(BaseTransform):
    pass

class Resample(BaseTransform):
    pass

class Slice(BaseTransform):
    pass