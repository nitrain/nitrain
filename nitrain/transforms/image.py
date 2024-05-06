
from .base import BaseTransform

__all__ = [
    'Astype',
    'Smooth',
    'Crop',
    'Resample',
    'Slice',
    'Pad',
    'PadLike'
]

class Astype(BaseTransform):
    """
    import ntimage as nti
    import nitrain as nt
    from nitrain import transforms as tx
    from nitrain.readers import MemoryReader
    dataset = nt.Dataset(
        inputs={'x':MemoryReader([nti.example('mni') for _ in range(10)]),
                'y':MemoryReader([nti.example('mni') for _ in range(10)])},
        outputs=MemoryReader([nti.example('mni') for _ in range(10)]),
        transforms={
            'x': tx.Astype('uint8')
        }
    )
    x, y = dataset[0]
    dataset2 = nt.Dataset(
        inputs=[MemoryReader([nti.example('mni') for _ in range(10)]),
                MemoryReader([nti.example('mni') for _ in range(10)])],
        outputs=MemoryReader([nti.example('mni') for _ in range(10)]),
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
    def __init__(self, sigma, method='gaussian'):
        self.sigma = sigma
        self.method = method
    
    def __call__(self, *images):
        images = [image.smooth(self.sigma, self.method) for image in images]
        return images if len(images) > 1 else images[0]

class Crop(BaseTransform):
    def __init__(self, indices=None):
        self.indices = indices
    def __call__(self, *images):
        images = [image.crop(self.indices) for image in images]
        return images if len(images) > 1 else images[0]


class Resample(BaseTransform):
    def __init__(self, shape, interpolation='linear', use_spacing=False):
        self.shape = shape
        self.interpolation = interpolation
        self.use_spacing = use_spacing
        
    def __call__(self, *images):
        images = [image.resample(self.shape,
                                 self.interpolation,
                                 self.use_spacing) for image in images]
        return images if len(images) > 1 else images[0]

class Slice(BaseTransform):
    def __init__(self, index, axis):
        self.axis = axis
        self.index = index
    
    def __call__(self, *images):
        images = [image.slice(self.index, self.axis) for image in images]
        return images if len(images) > 1 else images[0]


class Pad(BaseTransform):
    def __init__(self, width, mode='constant', **kwargs):
        self.width = width
        self.mode = mode
        self.kwargs = kwargs
    
    def __call__(self, *images):
        images = [image.pad(self.width, self.mode, **self.kwargs) for image in images]
        return images if len(images) > 1 else images[0]


class PadLike(BaseTransform):
    def __init__(self, target, mode='constant', **kwargs):
        self.target = target
        self.mode = mode
        self.kwargs = kwargs
    
    def __call__(self, *images):
        images = [image.pad_like(self.target, self.mode, **self.kwargs) for image in images]
        return images if len(images) > 1 else images[0]