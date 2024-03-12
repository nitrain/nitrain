
from .base_transform import BaseTransform

#("translation", "rigid", "scaleShear", "affine", "affineAndDeformation", "deformation")

class Affine(BaseTransform):
    pass

class Rotate(BaseTransform):
    pass

class Translate(BaseTransform):
    pass

class Shear(BaseTransform):
    pass

class Zoom(BaseTransform):
    pass

class Flip(BaseTransform):
    pass

class RandomDisplacementField(BaseTransform):
    pass
