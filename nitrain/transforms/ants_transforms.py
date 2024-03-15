# Transforms that apply specific antspy functions to images
import ants


import random
import numpy as np

from .base_transform import BaseTransform


class ApplyAntsTransform(BaseTransform):
    """
    Apply an ANTs transform to an image
    """
    def __init__(self, transform, reference=None):
        self.transform = transform
        self.reference = reference
        
    def __call__(self, image):
        reference = self.reference
        if reference is None:
            reference = image
            
        new_image = ants.apply_ants_transform_to_image(self.transform, image, reference)
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


class DisplacementField(BaseTransform):
    """
    Simulate and apply a random displacement field
    
    Examples
    --------
    >>> from nitrain import transforms as tx
    >>> import ants
    >>> img = ants.image_read(ants.get_data('r16'))
    >>> my_tx = tx.DisplacementField()
    >>> new_img = my_tx(img)
    >>> new_img.plot()
    """
    
    def __init__(self, 
                 field_type="bspline", 
                 number_of_random_points=1000, 
                 sd_noise=10.0,
                 enforce_stationary_boundary=True,
                 number_of_fitting_levels=4,
                 mesh_size=1,
                 sd_smoothing=4.0):
        self.field_type = field_type 
        self.number_of_random_points = number_of_random_points
        self.sd_noise = sd_noise
        self.enforce_stationary_boundary = enforce_stationary_boundary
        self.number_of_fitting_levels = number_of_fitting_levels
        self.mesh_size = mesh_size
        self.sd_smoothing = sd_smoothing
    
    def __call__(self, image):
        sim_field = ants.simulate_displacement_field(
            image,
            field_type = self.field_type,
            number_of_random_points = self.number_of_random_points,
            sd_noise = self.sd_noise,
            enforce_stationary_boundary = self.enforce_stationary_boundary,
            number_of_fitting_levels = self.number_of_fitting_levels,
            mesh_size = self.mesh_size,
            sd_smoothing = self.sd_smoothing
        )
        sim_transform = ants.transform_from_displacement_field(sim_field)
        new_image = ants.apply_ants_transform_to_image(sim_transform, image, image)
        return new_image


class BiasField(BaseTransform):
    """
    Simulate and apply a bias field
    
    Examples
    --------
    >>> from nitrain import transforms as tx
    >>> import ants
    >>> img = ants.image_read(ants.get_data('mni'))
    >>> my_tx = tx.BiasField()
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
        import antspynet
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
        if image.dimension == 2:
            transform_type = 'Euler2DTransform'
        elif image.dimension == 3:
            transform_type = 'Euler3DTransform'
        xfrm = ants.create_ants_transform(transform_type=transform_type, 
                                          center = center_of_mass_template,
                                          translation=translation,precision='float',
                                          dimension=image.dimension)

        new_image = ants.apply_ants_transform_to_image(xfrm, image, self.template)
        return new_image