
import math
import random
import numpy as np
import ants

from .base import BaseTransform

__all__ = [
    #'RandomShear',
    'RandomRotate',
    #'RandomZoom',
    #'RandomFlip',
    #'RandomTranslate',
]

class RandomShear(BaseTransform):
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

class RandomRotate(BaseTransform):
    def __init__(self, min_rotation, max_rotation, reference=None, p=1):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomRotate(-10, 10)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.RandomRotate((-90,0,0), (90,0,0), img)
        img2 = mytx(img)
        """
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.reference = reference
        self.p = p
        
        self.tx = ants.new_ants_transform(precision="float", 
                                          dimension=3 if isinstance(min_rotation, (tuple,list)) else 2,
                                          transform_type="AffineTransform")
        if self.reference is not None:
            self.tx.set_fixed_parameters(self.reference.get_center_of_mass())
        
    def __call__(self, *images):
        min_rotation = self.min_rotation
        max_rotation = self.max_rotation
        
        if random.uniform(0, 1) > self.p:
            return images if len(images) > 1 else images[0]
        
        if not isinstance(min_rotation, (tuple,list)):
            theta = math.pi / 180 * random.uniform(min_rotation, max_rotation)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], 
                                        [np.sin(theta), np.cos(theta), 0]])
        
        else:
            min_rotation_x, min_rotation_y, min_rotation_z = self.min_rotation
            max_rotation_x, max_rotation_y, max_rotation_z = self.max_rotation
            
            rotation_x = random.uniform(min_rotation_x, max_rotation_x)
            rotation_y = random.uniform(min_rotation_y, max_rotation_y)
            rotation_z = random.uniform(min_rotation_z, max_rotation_z)

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
        mytx = tx.Zoom((0.9, 0.9))
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Zoom((0.9, 0.9, 0.9))
        img2 = mytx(img)
        """
        if not isinstance(zoom, (list, tuple)):
            raise Exception("The zoom argument must be list or tuple.")

        self.zoom = zoom
        self.reference = reference

        self.tx = ants.new_ants_transform(
            precision="float", dimension=len(zoom), transform_type="AffineTransform"
        )
        if self.reference is not None:
            self.tx.set_fixed_parameters(self.reference.get_center_of_mass())

    def __call__(self, *images):
        zoom = self.zoom
        
        if len(zoom) == 2:
            zoom_matrix = np.array([[1/zoom[0], 0, 0], 
                                    [0, 1/zoom[1], 0]])
        elif len(zoom) == 3:
            zoom_matrix = np.array([[1/zoom[0], 0, 0, 0], 
                                    [0, 1/zoom[1], 0, 0],
                                    [0, 0, 1/zoom[2], 0]])
            
        self.tx.set_parameters(zoom_matrix)
        
        new_images = []
        for image in images:
            if not self.reference:
                self.tx.set_fixed_parameters(image.get_center_of_mass())
            new_image = self.tx.apply_to_image(image, self.reference)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]


class Translate(object):
    def __init__(self, translation, reference=None):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Translate((10, 30))
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Translate((30, 30,20))
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
        
        if len(translation) == 2:
            translation_matrix = np.array([[1, 0, translation[0]], 
                                    [0, 1, translation[1]]])
        elif len(translation) == 3:
            translation_matrix = np.array([[1, 0, 0, 0], 
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]])
            
        self.tx.set_parameters(translation_matrix)
        
        new_images = []
        for image in images:
            if not self.reference:
                self.tx.set_fixed_parameters(image.get_center_of_mass())
            new_image = self.tx.apply_to_image(image, self.reference)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]
    
class RandomFlip(BaseTransform):
    
    def __init__(self, axis=0):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Reflect()
        img2 = mytx(img)
        """
        self.axis = axis
        
    def __call__(self, *images):
        images = [ants.reflect_image(image, self.axis) for image in images]
        return images if len(images) > 1 else images[0]