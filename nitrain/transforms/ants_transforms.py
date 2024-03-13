# Transforms that apply specific antspy functions to images
import ants
import antspynet

import random
import numpy as np

from .base_transform import BaseTransform


class ApplyAntsTransform(BaseTransform):
    """
    Apply an ANTs transform to an image
    """
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, image):
        new_image = ants.apply_ants_transform_to_image(self.transform, image)
        return new_image


class BrainExtraction(BaseTransform):
    """
    Extract brain from image 
    """
    
    def __init__(self):
        pass
    
    def __call__(self, image):
        new_image = ants.get_mask(image)
        return new_image * image


class SimulateBiasField(BaseTransform):
    """
    Simulate and apply a bias field
    
    Examples
    --------
    >>> from nitrain import transforms as tx
    >>> import ants
    >>> img = ants.image_read(ants.get_data('mni'))
    >>> my_tx = tx.SimulateBiasField()
    >>> new_img = my_tx(img)
    >>> ants.plot_ortho_stack([new_img, img])
    """
    def __init__(self, sd=1.0, n_points=10, n_levels=2, mesh_size=10, field_power=2, normalize=False):
        self.sd = sd
        self.n_points = n_points
        self.n_levels = n_levels
        self.mesh_size = mesh_size
        self.field_power = field_power
        self.normalize = normalize
    
    def __call__(self, image):
        log_field = antspynet.simulate_bias_field(image, 
                                                  number_of_points=self.n_points, 
                                                  sd_bias_field=self.sd, 
                                                  number_of_fitting_levels=self.n_levels, 
                                                  mesh_size=self.mesh_size)
        log_field = log_field.iMath("Normalize")
        field_array = np.power(np.exp(log_field.numpy()), self.field_power)
        new_image = ants.image_clone(image) * ants.from_numpy(field_array, origin=image.origin, spacing=image.spacing, direction=image.direction)
        
        if self.normalize:
            new_image = (new_image - new_image.min()) / (new_image.max() - new_image.min())
            
        return new_image


class AlignWithTemplate(BaseTransform):
    """
    This function makes sure the images have the same
    orientation as a template.
    """
    def __init__(self, template):
        self.template = template

    def __call__(self, image):
        center_of_mass_template = ants.get_center_of_mass(self.template)
        center_of_mass_image = ants.get_center_of_mass(image)
        translation = tuple(np.array(center_of_mass_image) - np.array(center_of_mass_template))
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform", 
                                          center = center_of_mass_template,
                                          translation=translation,precision='float',
                                          dimension=image.dimension)

        new_image = ants.apply_ants_transform_to_image(xfrm, image, self.template)
        return new_image