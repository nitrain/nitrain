import ntimage as nt
import numpy as np
import random

from .base_transform import BaseTransform

class RandomAffine(BaseTransform):
    pass

class RandomRotate(BaseTransform):
    pass

class RandomTranslate(BaseTransform):
    """
    Examples
    --------
    >>> import ntimage as nt
    >>> from nitrain import transforms as tx
    >>> img = nt.load(nt.example_data('r16'))
    >>> my_tx = RandomTranslate(-20, 20)
    >>> img_tx = my_tx(img)
    >>> img_tx.plot(img)
    >>> img = nt.load(nt.example_data('mni'))
    >>> my_tx = RandomTranslate(-20, 20)
    >>> img_tx = my_tx(img)
    >>> img_tx.plot(img)
    """
    
    def __init__(self, min_value, max_value, reference=None):
        self.min_value = min_value
        self.max_value = max_value
        self.reference = reference
        if self.reference is not None:
            self.reference_com = self.reference.get_center_of_mass()
    
    def __call__(self, x, y=None):
        image_dim = x.dimension
        
        # create transform
        my_tx = nt.empty_transform(precision="float", 
                                      dimension=image_dim, 
                                      transform_type="AffineTransform")
        if self.reference is not None:
            my_tx.set_fixed_parameters(self.reference_com)
        
        # sample translation value
        min_value = self.min_value
        if isinstance(min_value, (int, float)):
            min_value = [min_value for _ in range(image_dim)]

        max_value = self.max_value
        if isinstance(max_value, (int, float)):
            max_value = [max_value for _ in range(image_dim)]
        
        tx_values = [random.uniform(min_value[i], max_value[i]) for i in range(image_dim)]
        
        if image_dim == 2:
            tx_matrix = np.array([[1, 0, tx_values[0]], 
                                  [0, 1, tx_values[1]]])
        elif image_dim == 3:
            tx_matrix = np.array([[1, 0, 0, tx_values[0]], 
                                  [0, 1, 0, tx_values[1]], 
                                  [0, 0, 1, tx_values[2]]])
            
        my_tx.set_parameters(tx_matrix)
        if y is None:
            return my_tx.apply_to_image(x, reference=self.reference)
        else:
            return (
                my_tx.apply_to_image(x, reference=self.reference),
                my_tx.apply_to_image(y, reference=self.reference),
            )
        

class RandomShear(BaseTransform):
    pass

class RandomZoom(BaseTransform):
    """
    Apply a random zoom transform to an image
    
    Examples
    --------
    >>> import ntimage as nt
    >>> from nitrain import transforms as tx
    >>> image = nt.load(nt.example_data('r16'))
    >>> my_tx = tx.RandomZoom(0.8,1.2)
    >>> new_image = my_tx(image)
    """
    
    def __init__(self, min_zoom, max_zoom):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
    
    def __call__(self, image):
        zoom = random.uniform(self.min_zoom, self.max_zoom)
        if image.dimension == 2:
            matrix =  np.array([[zoom, 0], [0, zoom]])
        else:
            matrix =  np.array([[zoom, 0, 0], [0, zoom, 0], [0, 0, zoom]])
        transform = create_centered_affine_transform(image, matrix)
        new_image = transform.apply_to_image(image)
        return new_image
        

class RandomFlip(BaseTransform):
    """
    Randomly flip an image with specified proabilikty.
    
    If no axis is specified, then the axis will be chosen
    randomly with equal probability.
    
    Examples
    --------
    >>> import ntimage as nt
    >>> from nitrain import transforms as tx
    >>> img = nt.load(nt.example_data('r16'))
    >>> my_tx = tx.RandomFlip(p=1)
    >>> new_img = my_tx(img)
    >>> new_img.plot()
    """
    def __init__(self, p=0.5, axis=None):
        self.p = p
        self.axis = axis
        
    def __call__(self, image):
        apply_tx = np.random.choice([True, False], size=1, p=[self.p, 1-self.p])[0]
        
        axis = self.axis
        if axis is None:
            axis = np.random.choice(range(image.dimension), size=1)[0]
            
        if apply_tx:
            image = image.reflect_image(axis=axis)
            
        return image
    
    def __repr__(self):
        return f'''tx.RandomFlip({self.p}, {self.axis})'''
        

def create_centered_affine_transform(image, matrix):
    transform = nt.empty_transform(
        transform_type="AffineTransform", 
        precision='float', 
        matrix=matrix,
        center=[image.shape[i]/2 for i in range(image.dimension)],
        dimension=image.dimension
    )
    return transform