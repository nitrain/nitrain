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

## DATA FUNCTIONS ##
def gray2d_setup():
    images = {}

    x = th.zeros(1,30,30)
    x[:,10:21,10:21] = 1
    images['square2d_gray'] = x
    return images

def color2d_setup():
    images = {}

    x = th.zeros(3,30,30)
    x[:,10:21,10:21] = 1
    images['square2d_color'] = x
    return images

def affine_setup():
    tforms = {}
    tforms['random_affine'] = RandomAffine(rotation_range=30, 
                                           translation_range=0.1)
    tforms['affine'] = Affine(th.FloatTensor([[0.9,0,0],[0,0.9,0]]))
    return tforms

def rotate_setup():
    tforms = {}
    tforms['random_rotate'] = RandomRotate(30)
    tforms['random_choice_rotate'] = RandomChoiceRotate([30,40,50])
    tforms['rotate'] = Rotate(30)
    return tforms

def translate_setup():
    tforms = {}
    tforms['random_translate'] = RandomTranslate(0.1)
    tforms['random_choice_translate'] = RandomChoiceTranslate([0.1,0.2])
    tforms['translate'] = Translate(0.3)
    return tforms

def shear_setup():
    tforms = {}
    tforms['random_shear'] = RandomShear(30)
    tforms['random_choice_shear'] = RandomChoiceShear([20,30,40])
    tforms['shear'] = Shear(25)
    return tforms

def zoom_setup():
    tforms = {}
    tforms['random_zoom'] = RandomZoom((0.8,1.2))
    tforms['random_choice_zoom'] = RandomChoiceZoom([0.8,0.9,1.1,1.2])
    tforms['zoom'] = Zoom(0.9)
    return tforms


def test_transforms():

    ### MAKE TRANSFORMS ###
    tforms = {}
    # AFFINE #
    tforms.update(affine_setup())

    # ROTATE #
    tforms.update(rotate_setup())

    # TRANSLATE #
    tforms.update(translate_setup())

    # SHEAR #
    tforms.update(shear_setup())

    # ZOOM #
    tforms.update(zoom_setup())

    ### MAKE DATA
    images = {}
    # 2d grayscale
    images.update(gray2d_setup())
    # 2d color
    images.update(color2d_setup())


    failures = []
    for im_key, im_val in images.items():
        for tf_key, tf_val in tforms.items():
            try:
                tf_val(im_val)
            except:
                failures.append((im_key, tf_key))

    print('FAILURES: ' , failures)


if __name__=='__main__':
    test_transforms()








