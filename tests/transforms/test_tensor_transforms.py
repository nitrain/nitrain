"""
Tests for torchsample/transforms/tensor_transforms.py
pytest tests/transforms/test_tensor_transforms.py -svvv --tb=line
"""

import torch as th
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(1, osp.join(cur_dir, '..'))
from torchsample.transforms import (ToTensor, ToCuda, ToFile, ChannelsLast, HWC, ChannelsFirst, CHW, TypeCast,
                                    AddChannel, Transpose, RangeNormalize, StdNormalize, RandomCrop, SpecialCrop, Pad,
                                    RandomFlip, RandomOrder)

# ----------------------------------------------------


class TestTensorTransforms(object):
    @classmethod
    def setup_class(cls):
        ## DATA SET ##
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
    def test_ToTensor(self):
        tforms = {}
        tforms['totensor'] = ToTensor()
        self._test_image_transforms_runtime(tforms)

    def ToCuda(self):
        tforms = {}
        tforms['tocuda'] = ToCuda()
        return tforms

    def test_ToFile(self):
        tforms = {}
        ROOT = '~/desktop/data/'
        tforms['tofile_npy'] = ToFile(root=ROOT, ext='.npy')
        tforms['tofile_pth'] = ToFile(root=ROOT, ext='.pth')
        tforms['tofile_jpg'] = ToFile(root=ROOT, ext='.jpg')
        tforms['tofile_png'] = ToFile(root=ROOT, ext='.png')
        self._test_image_transforms_runtime(tforms)

    def test_ChannelsLast(self):
        tforms = {}
        tforms['channels_last'] = ChannelsLast()
        tforms['hwc'] = HWC()
        self._test_image_transforms_runtime(tforms)

    def test_ChannelsFirst(self):
        tforms = {}
        tforms['channels_first'] = ChannelsFirst()
        tforms['chw'] = CHW()
        self._test_image_transforms_runtime(tforms)

    def test_TypeCast(self):
        tforms = {}
        tforms['byte'] = TypeCast('byte')
        tforms['double'] = TypeCast('double')
        tforms['float'] = TypeCast('float')
        tforms['int'] = TypeCast('int')
        tforms['long'] = TypeCast('long')
        tforms['short'] = TypeCast('short')
        self._test_image_transforms_runtime(tforms)

    def test_AddChannel(self):
        tforms = {}
        tforms['addchannel_axis0'] = AddChannel(axis=0)
        tforms['addchannel_axis1'] = AddChannel(axis=1)
        tforms['addchannel_axis2'] = AddChannel(axis=2)
        self._test_image_transforms_runtime(tforms)

    def test_Transpose(self):
        tforms = {}
        tforms['transpose_01'] = Transpose(0, 1)
        tforms['transpose_02'] = Transpose(0, 2)
        tforms['transpose_10'] = Transpose(1, 0)
        tforms['transpose_12'] = Transpose(1, 2)
        tforms['transpose_20'] = Transpose(2, 0)
        tforms['transpose_21'] = Transpose(2, 1)
        self._test_image_transforms_runtime(tforms)

    def test_RangeNormalize(self):
        tforms = {}
        tforms['rangenorm_01'] = RangeNormalize(0, 1)
        tforms['rangenorm_-11'] = RangeNormalize(-1, 1)
        tforms['rangenorm_-33'] = RangeNormalize(-3, 3)
        tforms['rangenorm_02'] = RangeNormalize(0, 2)
        self._test_image_transforms_runtime(tforms)

    def test_StdNormalize(self):
        tforms = {}
        tforms['stdnorm'] = StdNormalize()
        self._test_image_transforms_runtime(tforms)



    def test_RandomCrop(self):
        tforms = {}
        tforms['randomcrop_1010'] = RandomCrop((10, 10))
        tforms['randomcrop_510'] = RandomCrop((5, 10))
        tforms['randomcrop_105'] = RandomCrop((10, 5))
        tforms['randomcrop_99'] = RandomCrop((9, 9))
        tforms['randomcrop_79'] = RandomCrop((7, 9))
        tforms['randomcrop_97'] = RandomCrop((9, 7))
        self._test_image_transforms_runtime(tforms)

    def test_SpecialCrop(self):
        tforms = {}
        tforms['specialcrop_0_1010'] = SpecialCrop((10, 10), 0)
        tforms['specialcrop_0_510'] = SpecialCrop((5, 10), 0)
        tforms['specialcrop_0_105'] = SpecialCrop((10, 5), 0)
        tforms['specialcrop_0_99'] = SpecialCrop((9, 9), 0)
        tforms['specialcrop_0_79'] = SpecialCrop((7, 9), 0)
        tforms['specialcrop_0_97'] = SpecialCrop((9, 7), 0)

        tforms['specialcrop_1_1010'] = SpecialCrop((10, 10), 1)
        tforms['specialcrop_1_510'] = SpecialCrop((5, 10), 1)
        tforms['specialcrop_1_105'] = SpecialCrop((10, 5), 1)
        tforms['specialcrop_1_99'] = SpecialCrop((9, 9), 1)
        tforms['specialcrop_1_79'] = SpecialCrop((7, 9), 1)
        tforms['specialcrop_1_97'] = SpecialCrop((9, 7), 1)

        tforms['specialcrop_2_1010'] = SpecialCrop((10, 10), 2)
        tforms['specialcrop_2_510'] = SpecialCrop((5, 10), 2)
        tforms['specialcrop_2_105'] = SpecialCrop((10, 5), 2)
        tforms['specialcrop_2_99'] = SpecialCrop((9, 9), 2)
        tforms['specialcrop_2_79'] = SpecialCrop((7, 9), 2)
        tforms['specialcrop_2_97'] = SpecialCrop((9, 7), 2)

        tforms['specialcrop_3_1010'] = SpecialCrop((10, 10), 3)
        tforms['specialcrop_3_510'] = SpecialCrop((5, 10), 3)
        tforms['specialcrop_3_105'] = SpecialCrop((10, 5), 3)
        tforms['specialcrop_3_99'] = SpecialCrop((9, 9), 3)
        tforms['specialcrop_3_79'] = SpecialCrop((7, 9), 3)
        tforms['specialcrop_3_97'] = SpecialCrop((9, 7), 3)

        tforms['specialcrop_4_1010'] = SpecialCrop((10, 10), 4)
        tforms['specialcrop_4_510'] = SpecialCrop((5, 10), 4)
        tforms['specialcrop_4_105'] = SpecialCrop((10, 5), 4)
        tforms['specialcrop_4_99'] = SpecialCrop((9, 9), 4)
        tforms['specialcrop_4_79'] = SpecialCrop((7, 9), 4)
        tforms['specialcrop_4_97'] = SpecialCrop((9, 7), 4)
        self._test_image_transforms_runtime(tforms)


    def test_Pad(self):
        tforms = {}
        tforms['pad_4040'] = Pad((40, 40))
        tforms['pad_3040'] = Pad((30, 40))
        tforms['pad_4030'] = Pad((40, 30))
        tforms['pad_3939'] = Pad((39, 39))
        tforms['pad_3941'] = Pad((39, 41))
        tforms['pad_4139'] = Pad((41, 39))
        tforms['pad_4138'] = Pad((41, 38))
        tforms['pad_3841'] = Pad((38, 41))
        self._test_image_transforms_runtime(tforms)

    def test_RandomFlip(self):
        tforms = {}
        tforms['randomflip_h_01'] = RandomFlip(h=True, v=False)
        tforms['randomflip_h_02'] = RandomFlip(h=True, v=False, p=0)
        tforms['randomflip_h_03'] = RandomFlip(h=True, v=False, p=1)
        tforms['randomflip_h_04'] = RandomFlip(h=True, v=False, p=0.3)
        tforms['randomflip_v_01'] = RandomFlip(h=False, v=True)
        tforms['randomflip_v_02'] = RandomFlip(h=False, v=True, p=0)
        tforms['randomflip_v_03'] = RandomFlip(h=False, v=True, p=1)
        tforms['randomflip_v_04'] = RandomFlip(h=False, v=True, p=0.3)
        tforms['randomflip_hv_01'] = RandomFlip(h=True, v=True)
        tforms['randomflip_hv_02'] = RandomFlip(h=True, v=True, p=0)
        tforms['randomflip_hv_03'] = RandomFlip(h=True, v=True, p=1)
        tforms['randomflip_hv_04'] = RandomFlip(h=True, v=True, p=0.3)
        self._test_image_transforms_runtime(tforms)

    def test_RandomOrder(self):
        tforms = {}
        tforms['randomorder'] = RandomOrder()
        self._test_image_transforms_runtime(tforms)

    def _test_image_transforms_runtime(self, tforms):
        """
        Test that there are no runtime errors
        """
        successes = []
        failures = []
        for im_key, im_val in self.images.items():
            for tf_key, tf_val in tforms.items():
                try:
                    if isinstance(im_val, (tuple, list)):
                        if any(k in tf_key for k in ['randomflip', 'specialcrop', 'pad']):
                            continue
                        tf_val(*im_val)
                    else:
                        tf_val(im_val)
                    successes.append((im_key, tf_key))
                except:
                    if self.DEBUG:
                        if isinstance(im_val, (tuple, list)):
                            if any(k in tf_key for k in ['randomflip', 'specialcrop', 'pad']):
                                continue
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
