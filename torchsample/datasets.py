
from __future__ import absolute_import

import torch
import torch.utils.data

import os
import os.path
import warnings
import fnmatch

import numpy as np
try:
    import nibabel
except:
    warnings.warn('Cant import nibabel.. Cant load brain images')

try:
    from PIL import Image
except:
    warnings.warn('Cant import PIL.. Cant load PIL images')


def pil_loader(path):
    return Image.open(path).convert('RGB')

def npy_loader(path):
    return np.load(path)

def nifti_loader(path):
    return nibabel.load(path)

def _find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def _is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        '.nii.gz', '.npy'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def _finds_inputs_and_targets(directory, class_mode, class_to_idx=None, 
            input_regex=None, target_regex=None, ):
    """
    Map a dataset from a root folder
    """
    if class_mode == 'image':
        if not input_regex and not target_regex:
            raise ValueError('must give input_regex and target_regex if'+
                ' class_mode==image')
    inputs = []
    targets = []
    for subdir in sorted(os.listdir(directory)):
        d = os.path.join(directory, subdir)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if _is_image_file(fname):
                    if fnmatch.fnmatch(fname, input_regex):
                        path = os.path.join(root, fname)
                        inputs.append(path)
                        if class_mode == 'label':
                            targets.append(class_to_idx[subdir])
                    if class_mode == 'image' and \
                            fnmatch.fnmatch(fname, target_regex):
                        path = os.path.join(root, fname)
                        targets.append(path)
    if class_mode is None:
        return inputs
    else:
        return inputs, targets


class FolderDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 root,
                 class_mode='label',
                 input_regex='*',
                 target_regex=None,
                 transform=None, 
                 target_transform=None,
                 co_transform=None, 
                 file_loader='npy'):
        """
        Dataset class for loading out-of-memory data.

        Arguments
        ---------
        root : string
            path to main directory

        class_mode : string in `{'label', 'image'}`
            type of target sample to look for and return
            `label` = return class folder as target
            `image` = return another image as target as found by 'target_regex'
                NOTE: if class_mode == 'image', you must give an
                input and target regex and the input/target images should
                be in a folder together with no other images in that folder

        input_regex : string (default is any valid image file)
            regular expression to find input images
            e.g. if all your inputs have the word 'input', 
            you'd enter something like input_regex='*input*'
        
        target_regex : string (default is Nothing)
            regular expression to find target images if class_mode == 'image'
            e.g. if all your targets have the word 'segment', 
            you'd enter somthing like target_regex='*segment*'

        transform : torch transform
            transform to apply to input sample individually

        target_transform : torch transform
            transform to apply to target sample individually

        file_loader : string in `{'npy', 'pil', 'nifti'} or callable
            defines how to load samples from file
            if a function is provided, it should take in a file path
            as input and return the loaded sample.

        """
        if file_loader == 'npy':
            file_loader = npy_loader
        elif file_loader == 'pil':
            file_loader = pil_loader
        elif file_loader == 'nifti':
            file_loader = nifti_loader
        self.file_loader = file_loader

        root = os.path.expanduser(root)

        classes, class_to_idx = _find_classes(root)
        inputs, targets = _finds_inputs_and_targets(root, class_mode,
            class_to_idx, input_regex, target_regex)

        if len(inputs) == 0:
            raise(RuntimeError('Found 0 images in subfolders of: %s' % root))
        else:
            print('Found %i images' % len(inputs))

        self.root = os.path.expanduser(root)
        self.inputs = inputs
        self.targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        
        self.class_mode = class_mode

    def __getitem__(self, index):
        # get paths
        input_sample = self.inputs[index]
        target_sample = self.targets[index]

        # load samples into memory
        input_sample = torch.from_numpy(self.file_loader(input_sample)).contiguous()
        if self.class_mode == 'image':
            target_sample = torch.from_numpy(self.file_loader(target_sample)).contiguous()
        
        # apply transforms
        if self.transform is not None:
            input_sample = self.transform(input_sample)
        if self.target_transform is not None:
            target_sample = self.target_transform(target_sample)
        if self.co_transform is not None:
            input_sample, target_sample = self.co_transform(input_sample, target_sample)

        return input_sample, target_sample
    
    def __len__(self):
        return len(self.inputs)


class TensorDataset(torch.utils.data.Dataset):

    def __init__(self,
                 input_tensor,
                 target_tensor=None,
                 transform=None, 
                 target_transform=None,
                 co_transform=None):
        """
        Dataset class for loading in-memory data.

        Arguments
        ---------
        input_tensor : torch tensor

        target_tensor : torch tensor

        transform : torch transform
            transform to apply to input sample individually

        target_transform : torch transform
            transform to apply to target sample individually

        file_loader : string in `{'npy', 'pil', 'nifti'} or callable
            defines how to load samples from file
            if a function is provided, it should take in a file path
            as input and return the loaded sample.

        """
        self.inputs = input_tensor
        self.targets = target_tensor
        if target_tensor is None:
            self.has_target = False
        else:
            self.has_target = True
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform

    def __getitem__(self, index):
        """Return a (transformed) input and target sample from an integer index"""
        # get paths
        input_sample = self.inputs[index]
        if self.has_target:
            target_sample = self.targets[index]

        # apply transforms
        if self.transform is not None:
            input_sample = self.transform(input_sample)
        if self.has_target and self.target_transform is not None:
            target_sample = self.target_transform(target_sample)
        if self.has_target and self.co_transform is not None:
            input_sample, target_sample = self.co_transform(input_sample, target_sample)

        if self.has_target:
            return input_sample, target_sample
        else:
            return input_sample

    def __len__(self):
        """Number of samples"""
        return self.inputs.size(0)


