"""Tests for torchsample/transforms/image_transforms.py python -m pytest
tests/transforms/test_image_transforms.py -svvv --tb=line."""

import torch as th

from torchsample.transforms import (Grayscale, RandomGrayscale, Gamma, RandomGamma, RandomChoiceGamma, Brightness,
                                    RandomBrightness, RandomChoiceBrightness, Saturation, RandomSaturation,
                                    RandomChoiceSaturation, Contrast, RandomContrast, RandomChoiceContrast)

# ----------------------------------------------------
# ----------------------------------------------------


class TestImageTransforms(object):

    @classmethod
    def setup_class(cls):
        # the test img resolution is 400x300
        # cls.img_path = osp.join(osp.dirname(__file__), 'data/color.jpg')
        # cls.gray_img_path = osp.join(
        #     osp.dirname(__file__), 'data/grayscale.jpg')
        # cls.img = cv2.imread(cls.img_path)
        ### MAKE DATA ###
        images = {}

        cls.gray2d_images = cls.gray2d_setup()
        images.update(cls.gray2d_images)

        cls.multi_gray2d_images = cls.multi_gray2d_setup(cls)
        images.update(cls.multi_gray2d_images)

        cls.color2d_images = cls.color2d_setup()
        images.update(cls.color2d_images)

        cls.multi_color2d_images = cls.multi_color2d_setup(cls)
        images.update(cls.multi_color2d_images)

        cls.images = images
        cls.verbose = 0
        cls.DEBUG = True

    # def test_setup_class(self):
    #     print(self.images.keys())

    ## DATA SET ##
    @staticmethod
    def gray2d_setup():
        images = {}

        x = th.zeros(1, 30, 30)
        x[:, 10:21, 10:21] = 1
        images['gray_01'] = x

        x = th.zeros(1, 30, 40)
        x[:, 10:21, 10:21] = 1
        images['gray_02'] = x

        return images

    def multi_gray2d_setup(self):
        old_imgs = self.gray2d_setup()
        images = {}
        for k, v in old_imgs.items():
            images[k + '_2imgs'] = [v, v]
            images[k + '_3imgs'] = [v, v, v]
            images[k + '_4imgs'] = [v, v, v, v]
        return images

    @staticmethod
    def color2d_setup():
        images = {}

        x = th.zeros(3, 30, 30)
        x[:, 10:21, 10:21] = 1
        images['color_01'] = x

        x = th.zeros(3, 30, 40)
        x[:, 10:21, 10:21] = 1
        images['color_02'] = x

        return images

    def multi_color2d_setup(self):
        old_imgs = self.color2d_setup()
        images = {}
        for k, v in old_imgs.items():
            images[k + '_2imgs'] = [v, v]
            images[k + '_3imgs'] = [v, v, v]
            images[k + '_4imgs'] = [v, v, v, v]
        return images

    # ----------------------------------------------------
    # ----------------------------------------------------
    ## TFORMS SETUP ###

    def test_Grayscale(self):
        tforms = {}
        tforms['grayscale_keepchannels'] = Grayscale(keep_channels=True)
        tforms['grayscale_dontkeepchannels'] = Grayscale(keep_channels=False)
        self._test_image_transforms_runtime(tforms)

    def test_RandomGrayscale(self):
        tforms = dict(
            random_grayscale_nop=RandomGrayscale(),
            random_grayscale_p_01=RandomGrayscale(0),
            random_grayscale_p_02=RandomGrayscale(0.5),
            random_grayscale_p_03=RandomGrayscale(1),
        )
        self._test_image_transforms_runtime(tforms)

    def test_Gamma(self):
        tforms = {}
        tforms['gamma_<1'] = Gamma(value=0.5)
        tforms['gamma_=1'] = Gamma(value=1.0)
        tforms['gamma_>1'] = Gamma(value=1.5)
        self._test_image_transforms_runtime(tforms)

    def test_RandomGamma(self):
        tforms = {}
        tforms['random_gamma_01'] = RandomGamma(0.5, 1.5)
        tforms['random_gamma_02'] = RandomGamma(0.5, 1.0)
        tforms['random_gamma_03'] = RandomGamma(1.0, 1.5)
        self._test_image_transforms_runtime(tforms)

    def test_RandomChoiceGamma(self):
        tforms = {}
        tforms['random_choice_gamma_01'] = RandomChoiceGamma([0.5, 1.0])
        tforms['random_choice_gamma_02'] = RandomChoiceGamma([0.5, 1.0], p=[0.5, 0.5])
        tforms['random_choice_gamma_03'] = RandomChoiceGamma([0.5, 1.0], p=[0.2, 0.8])
        self._test_image_transforms_runtime(tforms)

    def test_Brightness(self):
        tforms = {}
        tforms['brightness_=-1'] = Brightness(value=-1)
        tforms['brightness_<0'] = Brightness(value=-0.5)
        tforms['brightness_=0'] = Brightness(value=0)
        tforms['brightness_>0'] = Brightness(value=0.5)
        tforms['brightness_=1'] = Brightness(value=1)
        self._test_image_transforms_runtime(tforms)

    def test_RandomBrightness(self):
        tforms = {}
        tforms['random_brightness_01'] = RandomBrightness(-1, -0.5)
        tforms['random_brightness_02'] = RandomBrightness(-0.5, 0)
        tforms['random_brightness_03'] = RandomBrightness(0, 0.5)
        tforms['random_brightness_04'] = RandomBrightness(0.5, 1)
        self._test_image_transforms_runtime(tforms)

    def test_RandomChoiceBrightness(self):
        tforms = {}
        tforms['random_choice_brightness_01'] = RandomChoiceBrightness([-1, 0, 1])
        tforms['random_choice_brightness_02'] = RandomChoiceBrightness([-1, 0, 1], p=[0.2, 0.5, 0.3])
        tforms['random_choice_brightness_03'] = RandomChoiceBrightness([0, 0, 0, 0], p=[0.25, 0.5, 0.125, 0.125])
        self._test_image_transforms_runtime(tforms)

    def test_Saturation(self,):
        tforms = {}
        tforms['saturation_=-1'] = Saturation(-1)
        tforms['saturation_<0'] = Saturation(-0.5)
        tforms['saturation_=0'] = Saturation(0)
        tforms['saturation_>0'] = Saturation(0.5)
        tforms['saturation_=1'] = Saturation(1)
        self._test_image_transforms_runtime(tforms)

    def test_RandomSaturation(self):
        tforms = {}
        tforms['random_saturation_01'] = RandomSaturation(-1, -0.5)
        tforms['random_saturation_02'] = RandomSaturation(-0.5, 0)
        tforms['random_saturation_03'] = RandomSaturation(0, 0.5)
        tforms['random_saturation_04'] = RandomSaturation(0.5, 1)
        self._test_image_transforms_runtime(tforms)

    def test_RandomChoiceSaturation(self):
        tforms = {}
        tforms['random_choice_saturation_01'] = RandomChoiceSaturation([-1, 0, 1])
        tforms['random_choice_saturation_02'] = RandomChoiceSaturation([-1, 0, 1], p=[0.2, 0.5, 0.3])
        tforms['random_choice_saturation_03'] = RandomChoiceSaturation([0, 0, 0, 0], p=[0.125, 0.5, 0.125, 0.25])
        self._test_image_transforms_runtime(tforms)

    def test_Contrast(self):
        tforms = {}
        tforms['contrast_<<0'] = Contrast(-10)
        tforms['contrast_<0'] = Contrast(-1)
        tforms['contrast_=0'] = Contrast(0)
        tforms['contrast_>0'] = Contrast(1)
        tforms['contrast_>>0'] = Contrast(10)
        self._test_image_transforms_runtime(tforms)

    def test_RandomContrast(self):
        tforms = {}
        tforms['random_contrast_01'] = RandomContrast(-10, -1)
        tforms['random_contrast_02'] = RandomContrast(-1, 0)
        tforms['random_contrast_03'] = RandomContrast(0, 1)
        tforms['random_contrast_04'] = RandomContrast(1, 10)
        self._test_image_transforms_runtime(tforms)

    def test_RandomChoiceContrast(self):
        tforms = {}
        tforms['random_choice_saturation_01'] = RandomChoiceContrast([-1, 0, 1])
        tforms['random_choice_saturation_02'] = RandomChoiceContrast([-10, 0, 10], p=[0.2, 0.5, 0.3])
        tforms['random_choice_saturation_03'] = RandomChoiceContrast([1, 1], p=[0.5, 0.5])
        self._test_image_transforms_runtime(tforms)

    def _test_image_transforms_runtime(self, tforms):
        """Test that there are no runtime errors."""
        successes = []
        failures = []

        for im_key, im_val in self.images.items():
            for tf_key, tf_val in tforms.items():
                try:
                    if isinstance(im_val, (tuple, list)):
                        tf_val(*im_val)
                    else:
                        tf_val(im_val)
                    successes.append((im_key, tf_key))
                except:
                    if self.DEBUG:
                        if isinstance(im_val, (tuple, list)):
                            tf_val(*im_val)
                        else:
                            tf_val(im_val)
                        raise RuntimeError('{} {} {}'.format(im_key, tf_key, tf_val))
                    failures.append((im_key, tf_key))

        if self.verbose > 0:
            for k, v in failures:
                print('%s - %s' % (k, v))
            print()
            print('# SUCCESSES: ', len(successes))
            print('# FAILURES: ', len(failures))
        assert len(failures) == 0


# if __name__ == '__main__':
#     test_image_transforms_runtime()
