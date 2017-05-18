"""
Tests for torchsample/transforms/image_transforms.py
"""


import torch as th

from torchsample.transforms import (Grayscale, RandomGrayscale,
                    Gamma, RandomGamma, RandomChoiceGamma,
                    Brightness, RandomBrightness, RandomChoiceBrightness,
                    Saturation, RandomSaturation, RandomChoiceSaturation,
                    Contrast, RandomContrast, RandomChoiceContrast)

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

## TFORMS SETUP ###
def Grayscale_setup():
    tforms = {}
    tforms['grayscale_keepchannels'] = Grayscale(keep_channels=True)
    tforms['grayscale_dontkeepchannels'] = Grayscale(keep_channels=False)

    tforms['random_grayscale_nop'] = RandomGrayscale()
    tforms['random_grayscale_p_01'] = RandomGrayscale(0)
    tforms['random_grayscale_p_02'] = RandomGrayscale(0.5)
    tforms['random_grayscale_p_03'] = RandomGrayscale(1)

    return tforms

def Gamma_setup():
    tforms = {}
    tforms['gamma_<1'] = Gamma(value=0.5)
    tforms['gamma_=1'] = Gamma(value=1.0)
    tforms['gamma_>1'] = Gamma(value=1.5)
    tforms['random_gamma_01'] = RandomGamma(0.5,1.5)
    tforms['random_gamma_02'] = RandomGamma(0.5,1.0)
    tforms['random_gamma_03'] = RandomGamma(1.0,1.5)
    tforms['random_choice_gamma_01'] = RandomChoiceGamma([0.5,1.0])
    tforms['random_choice_gamma_02'] = RandomChoiceGamma([0.5,1.0],p=[0.5,0.5])
    tforms['random_choice_gamma_03'] = RandomChoiceGamma([0.5,1.0],p=[0.2,0.8])

    return tforms

def Brightness_setup():
    tforms = {}
    tforms['brightness_=-1'] = Brightness(value=-1)
    tforms['brightness_<0'] = Brightness(value=-0.5)
    tforms['brightness_=0'] = Brightness(value=0)
    tforms['brightness_>0'] = Brightness(value=0.5)
    tforms['brightness_=1'] = Brightness(value=1)

    tforms['random_brightness_01'] = RandomBrightness(-1,-0.5)
    tforms['random_brightness_02'] = RandomBrightness(-0.5,0)
    tforms['random_brightness_03'] = RandomBrightness(0,0.5)
    tforms['random_brightness_04'] = RandomBrightness(0.5,1)

    tforms['random_choice_brightness_01'] = RandomChoiceBrightness([-1,0,1])
    tforms['random_choice_brightness_02'] = RandomChoiceBrightness([-1,0,1],p=[0.2,0.5,0.3])
    tforms['random_choice_brightness_03'] = RandomChoiceBrightness([0,0,0,0],p=[0.25,0.5,0.25,0.25])

    return tforms

def Saturation_setup():
    tforms = {}
    tforms['saturation_=-1'] = Saturation(-1)
    tforms['saturation_<0'] = Saturation(-0.5)
    tforms['saturation_=0'] = Saturation(0)
    tforms['saturation_>0'] = Saturation(0.5)
    tforms['saturation_=1'] = Saturation(1)

    tforms['random_saturation_01'] = RandomSaturation(-1,-0.5)
    tforms['random_saturation_02'] = RandomSaturation(-0.5,0)
    tforms['random_saturation_03'] = RandomSaturation(0,0.5)
    tforms['random_saturation_04'] = RandomSaturation(0.5,1)

    tforms['random_choice_saturation_01'] = RandomChoiceSaturation([-1,0,1])
    tforms['random_choice_saturation_02'] = RandomChoiceSaturation([-1,0,1],p=[0.2,0.5,0.3])
    tforms['random_choice_saturation_03'] = RandomChoiceSaturation([0,0,0,0],p=[0.25,0.5,0.25,0.25])
    
    return tforms

def Contrast_setup():
    tforms = {}
    tforms['contrast_<<0'] = Contrast(-10)
    tforms['contrast_<0'] = Contrast(-1)
    tforms['contrast_=0'] = Contrast(0)
    tforms['contrast_>0'] = Contrast(1)
    tforms['contrast_>>0'] = Contrast(10)

    tforms['random_contrast_01'] = RandomContrast(-10,-1)
    tforms['random_contrast_02'] = RandomContrast(-1,0)
    tforms['random_contrast_03'] = RandomContrast(0,1)
    tforms['random_contrast_04'] = RandomContrast(1,10)

    tforms['random_choice_saturation_01'] = RandomChoiceContrast([-1,0,1])
    tforms['random_choice_saturation_02'] = RandomChoiceContrast([-10,0,10],p=[0.2,0.5,0.3])
    tforms['random_choice_saturation_03'] = RandomChoiceContrast([1,1],p=[0.5,0.5])
    
    return tforms

# ----------------------------------------------------
# ----------------------------------------------------

def test_image_transforms_runtime(verbose=1):
    """
    Test that there are no runtime errors
    """
    ### MAKE TRANSFORMS ###
    tforms = {}
    tforms.update(Gamma_setup())
    tforms.update(Brightness_setup())
    tforms.update(Saturation_setup())
    tforms.update(Contrast_setup())

    ### MAKE DATA ###
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
    test_image_transforms_runtime()
