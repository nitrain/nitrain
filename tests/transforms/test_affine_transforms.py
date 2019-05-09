"""
Test affine transforms
python -m pytest tests/transforms/test_affine_transforms.py -svvv --tb=line
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

from torchsample.transforms import (RandomAffine, Affine, RandomRotate, RandomChoiceRotate, Rotate, RandomTranslate,
                                    RandomChoiceTranslate, Translate, RandomShear, RandomChoiceShear, Shear, RandomZoom,
                                    RandomChoiceZoom, Zoom)

# ----------------------------------------------------
# ----------------------------------------------------


class TestAffineTransforms(object):

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

    def test_Affine(self):
        tforms = {}
        tforms['random_affine'] = RandomAffine(rotation_range=30, translation_range=0.1)
        tforms['affine'] = Affine(th.FloatTensor([[0.9, 0, 0], [0, 0.9, 0]]))
        self._test_affine_transforms_runtime(tforms)

    def test_Rotate(self):
        tforms = {}
        tforms['random_rotate'] = RandomRotate(30)
        tforms['random_choice_rotate'] = RandomChoiceRotate([30, 40, 50])
        tforms['rotate'] = Rotate(30)
        self._test_affine_transforms_runtime(tforms)

    def test_Translate(self):
        tforms = {}
        tforms['random_translate'] = RandomTranslate(0.1)
        tforms['random_choice_translate'] = RandomChoiceTranslate([0.1, 0.2])
        tforms['translate'] = Translate(0.3)
        self._test_affine_transforms_runtime(tforms)

    def test_Shear(self):
        tforms = {}
        tforms['random_shear'] = RandomShear(30)
        tforms['random_choice_shear'] = RandomChoiceShear([20, 30, 40])
        tforms['shear'] = Shear(25)
        self._test_affine_transforms_runtime(tforms)

    def test_Zoom(self):
        tforms = {}
        tforms['random_zoom'] = RandomZoom((0.8, 1.2))
        tforms['random_choice_zoom'] = RandomChoiceZoom([0.8, 0.9, 1.1, 1.2])
        tforms['zoom'] = Zoom(0.9)
        self._test_affine_transforms_runtime(tforms)

    # ----------------------------------------------------
    # ----------------------------------------------------
    def _test_affine_transforms_runtime(self, tforms):
        """
        Test that there are no runtime errors
        """
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
#     test_affine_transforms_runtime()
