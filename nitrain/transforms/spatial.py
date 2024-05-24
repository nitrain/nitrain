import ants
import math
import numpy as np

from .base import BaseTransform

__all__ = [
    'ApplyAntsTransform',
    'AffineTransform',
    'Shear',
    'Rotate',
    'Zoom',
    'Flip',
    'Translate'
]

class ApplyAntsTransform(BaseTransform):
    
    def __init__(self, transform):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        ants_tx = ants.new_ants_transform(dimension=2)
        ants_tx.set_parameters(ants_tx.parameters*2)
        mytx = tx.ApplyAntsTransform(ants_tx)
        img2 = mytx(img)
        """
        self.transform = transform

    def __call__(self, *images):
        images = [self.transform.apply_to_image(image) for image in images]
        return images if len(images) > 1 else images[0]
    
class AffineTransform(BaseTransform):
    def __init__(self, array, reference=None):
        """
        import ants
        import math
        import numpy as np
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        theta = math.radians(90)
        arr = np.array([[math.cos(theta),-math.sin(theta), 0],
                        [math.sin(theta),math.cos(theta), 0]])
        mytx = tx.AffineTransform(arr)
        img2 = mytx(img)
        """
        self.array = array
        self.reference = reference
            
        self.tx = ants.new_ants_transform(precision="float", 
                                          dimension=array.shape[0], 
                                          transform_type="AffineTransform")
        if self.reference is not None:
            self.tx.set_fixed_parameters(self.reference.get_center_of_mass())
        
    def __call__(self, *images):
        self.tx.set_parameters(self.array)
        new_images = []
        for image in images:
            if not self.reference:
                self.tx.set_fixed_parameters(image.get_center_of_mass())
            new_image = self.tx.apply_to_image(image, self.reference)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]

class Shear(BaseTransform):
    def __init__(self, shear, reference=None):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Shear((0,10))
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Shear((0,10,0))
        img2 = mytx(img)
        """
        if not isinstance(shear, (list, tuple)):
            raise Exception('The shear arg must be list or tuple with length equal to image dimension.')
        
        self.shear = shear
        self.reference = reference
            
        self.tx = ants.new_ants_transform(precision="float", 
                                          dimension=len(shear),
                                          transform_type="AffineTransform")
        if self.reference is not None:
            self.tx.set_fixed_parameters(self.reference.get_center_of_mass())
        
    def __call__(self, *images):
        shear = [math.pi / 180 * s for s in self.shear]
        if len(shear) == 2:
            shear_matrix = np.array([[1, shear[0], 0], [shear[1], 1, 0]])
        elif len(shear) == 3:
            shear_matrix = np.array([[1, shear[0], shear[0], 0],
                                     [shear[1], 1, shear[1], 0],
                                     [shear[2], shear[2], 1, 0]])
        self.tx.set_parameters(shear_matrix)
        
        new_images = []
        for image in images:
            if not self.reference:
                self.tx.set_fixed_parameters(image.get_center_of_mass())
            new_image = self.tx.apply_to_image(image, self.reference)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]

class Rotate(BaseTransform):
    def __init__(self, rotation, reference=None):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Rotate(10)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Rotate((90,0,0), img)
        img2 = mytx(img)
        """
        self.rotation = rotation
        self.reference = reference
            
        self.tx = ants.new_ants_transform(precision="float", 
                                          dimension=3 if isinstance(rotation, (tuple,list)) else 2,
                                          transform_type="AffineTransform")
        if self.reference is not None:
            self.tx.set_fixed_parameters(self.reference.get_center_of_mass())
        
    def __call__(self, *images):
        rotation = self.rotation
        if not isinstance(rotation, (tuple,list)):
            theta = math.pi / 180 * rotation
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], 
                                        [np.sin(theta), np.cos(theta), 0]])
        
        else:
            rotation_x, rotation_y, rotation_z = self.rotation

            # Rotation about X axis
            theta_x = math.pi / 180 * rotation_x
            rotate_matrix_x = np.array(
                [
                    [1, 0, 0, 0],
                    [0, math.cos(theta_x), -math.sin(theta_x), 0],
                    [0, math.sin(theta_x), math.cos(theta_x), 0],
                    [0, 0, 0, 1],
                ]
            )

            # Rotation about Y axis
            theta_y = math.pi / 180 * rotation_y
            rotate_matrix_y = np.array(
                [
                    [math.cos(theta_y), 0, math.sin(theta_y), 0],
                    [0, 1, 0, 0],
                    [-math.sin(theta_y), 0, math.cos(theta_y), 0],
                    [0, 0, 0, 1],
                ]
            )

            # Rotation about Z axis
            theta_z = math.pi / 180 * rotation_z
            rotate_matrix_z = np.array(
                [
                    [math.cos(theta_z), -math.sin(theta_z), 0, 0],
                    [math.sin(theta_z), math.cos(theta_z), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            rotation_matrix = rotate_matrix_x.dot(rotate_matrix_y).dot(rotate_matrix_z)[:3, :]
        
        self.tx.set_parameters(rotation_matrix)
        
        new_images = []
        for image in images:
            if not self.reference:
                self.tx.set_fixed_parameters(image.get_center_of_mass())
            new_image = self.tx.apply_to_image(image, self.reference)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]

class Zoom(object):
    def __init__(self, zoom, reference=None):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Zoom(0.9)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Zoom(0.9)
        img2 = mytx(img)
        """
        self.zoom = zoom
        self.reference = reference

    def __call__(self, *images):        
        new_images = []
        for image in images:
            nd = image.dimension
            
            tx = ants.new_ants_transform(
                precision="float", dimension=nd, transform_type="AffineTransform"
            )
            if self.reference:
                tx.set_fixed_parameters(self.reference.get_center_of_mass())
            else:
                tx.set_fixed_parameters(image.get_center_of_mass())
                
            zoom = self.zoom
            zoom_matrix = np.concatenate((np.eye(nd)*1/zoom, np.zeros((nd,1))), axis=1)
            tx.set_parameters(zoom_matrix)
        
            new_image = tx.apply_to_image(image, self.reference)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]

class Flip(BaseTransform):
    
    def __init__(self, axis=0):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Flip()
        img2 = mytx(img)
        """
        self.axis = axis
        
    def __call__(self, *images):
        images = [ants.reflect_image(image, self.axis) for image in images]
        return images if len(images) > 1 else images[0]

class Translate(object):
    def __init__(self, translation, reference=None):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Translate((10, 30))
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Translate((30, 30, 20))
        img2 = mytx(img)
        """
        if not isinstance(translation, (list, tuple)):
            raise Exception("The translation argument must be list or tuple.")

        self.translation = translation
        self.reference = reference

        self.tx = ants.new_ants_transform(
            precision="float", dimension=len(translation), transform_type="AffineTransform"
        )
        if self.reference is not None:
            self.tx.set_fixed_parameters(self.reference.get_center_of_mass())

    def __call__(self, *images):
        translation = self.translation
        
        nd = len(translation)
        translation_matrix = np.concatenate((np.eye(nd), 
                                             np.array(translation).reshape(nd,1)), 
                                            axis=1)
            
        self.tx.set_parameters(translation_matrix)
        
        new_images = []
        for image in images:
            if not self.reference:
                self.tx.set_fixed_parameters(image.get_center_of_mass())
            new_image = self.tx.apply_to_image(image, self.reference)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]
