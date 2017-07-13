"""
Test affine transforms

Transforms:
    - Affine + RandomAffine
    - AffineCompose
    - Rotate + RandomRotate
    - Translate + RandomTranslate
    - Shear + RandomShear
    - Zoom + RandomZoom
"""

#import pytest

import torch as th

from torchsample.transforms import (RandomAffine, Affine,
                        RandomRotate, RandomChoiceRotate, Rotate,
                        RandomTranslate, RandomChoiceTranslate, Translate,
                        RandomShear, RandomChoiceShear, Shear,
                        RandomZoom, RandomChoiceZoom, Zoom)

# ----------------------------------------------------
# ----------------------------------------------------

## DATA SET ##
def gray2d_setup():
    images = {}

    x = th.zeros(1,30,30)
    x[:,10:21,10:21] = 1
    images['gray_01'] = x

    x = th.zeros(1,30,40)
    x[:,10:21,10:21] = 1
    images['gray_02'] = x

    return images

def multi_gray2d_setup():
    old_imgs = gray2d_setup()
    images = {}
    for k,v in old_imgs.items():
        images[k+'_2imgs'] = [v,v]
        images[k+'_3imgs'] = [v,v,v]
        images[k+'_4imgs'] = [v,v,v,v]
    return images

def color2d_setup():
    images = {}

    x = th.zeros(3,30,30)
    x[:,10:21,10:21] = 1
    images['color_01'] = x

    x = th.zeros(3,30,40)
    x[:,10:21,10:21] = 1
    images['color_02'] = x

    return images

def multi_color2d_setup():
    old_imgs = color2d_setup()
    images = {}
    for k,v in old_imgs.items():
        images[k+'_2imgs'] = [v,v]
        images[k+'_3imgs'] = [v,v,v]
        images[k+'_4imgs'] = [v,v,v,v]
    return images


# ----------------------------------------------------
# ----------------------------------------------------


def Affine_setup():
    tforms = {}
    tforms['random_affine'] = RandomAffine(rotation_range=30, 
                                           translation_range=0.1)
    tforms['affine'] = Affine(th.FloatTensor([[0.9,0,0],[0,0.9,0]]))
    return tforms

def Rotate_setup():
    tforms = {}
    tforms['random_rotate'] = RandomRotate(30)
    tforms['random_choice_rotate'] = RandomChoiceRotate([30,40,50])
    tforms['rotate'] = Rotate(30)
    return tforms

def Translate_setup():
    tforms = {}
    tforms['random_translate'] = RandomTranslate(0.1)
    tforms['random_choice_translate'] = RandomChoiceTranslate([0.1,0.2])
    tforms['translate'] = Translate(0.3)
    return tforms

def Shear_setup():
    tforms = {}
    tforms['random_shear'] = RandomShear(30)
    tforms['random_choice_shear'] = RandomChoiceShear([20,30,40])
    tforms['shear'] = Shear(25)
    return tforms

def Zoom_setup():
    tforms = {}
    tforms['random_zoom'] = RandomZoom((0.8,1.2))
    tforms['random_choice_zoom'] = RandomChoiceZoom([0.8,0.9,1.1,1.2])
    tforms['zoom'] = Zoom(0.9)
    return tforms

# ----------------------------------------------------
# ----------------------------------------------------

def test_affine_transforms_runtime(verbose=1):
    """
    Test that there are no runtime errors
    """
    ### MAKE TRANSFORMS ###
    tforms = {}
    tforms.update(Affine_setup())
    tforms.update(Rotate_setup())
    tforms.update(Translate_setup())
    tforms.update(Shear_setup())
    tforms.update(Zoom_setup())

    ### MAKE DATA
    images = {}
    images.update(gray2d_setup())
    images.update(multi_gray2d_setup())
    images.update(color2d_setup())
    images.update(multi_color2d_setup())

    successes = []
    failures = []
    for im_key, im_val in images.items():
        for tf_key, tf_val in tforms.items():
            try:
                if isinstance(im_val, (tuple,list)):
                    tf_val(*im_val)
                else:
                    tf_val(im_val)
                successes.append((im_key, tf_key))
            except:
                failures.append((im_key, tf_key))

    if verbose > 0:
        for k, v in failures:
            print('%s - %s' % (k, v))

    print('# SUCCESSES: ', len(successes))
    print('# FAILURES: ' , len(failures))


if __name__=='__main__':
    test_affine_transforms_runtime()








