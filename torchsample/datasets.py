
from __future__ import absolute_import

from .dataset_iter import default_collate, DatasetIter
from .samplers import RandomSampler, SequentialSampler

import torch

import os
import os.path
import warnings
import fnmatch
import math

import numpy as np
try:
    import nibabel
except:
    warnings.warn('Cant import nibabel.. Cant load brain images')

try:
    from PIL import Image
except:
    warnings.warn('Cant import PIL.. Cant load PIL images')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.nii.gz', '.npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def pil_loader(path):
    return Image.open(path).convert('RGB')

def npy_loader(path):
    return torch.from_numpy(np.load(path).astype('float32'))

def nifti_loader(path):
    return nibabel.load(path)

def make_dataset(directory, class_mode, class_to_idx=None, 
            input_regex=None, target_regex=None, ):
    """Map a dataset from a root folder"""
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
                if fnmatch.fnmatch(fname, input_regex):
                    path = os.path.join(root, fname)
                    inputs.append(path)
                    if class_mode == 'label':
                        targets.append(class_to_idx[subdir])
                if class_mode == 'image' and fnmatch.fnmatch(fname, target_regex):
                    path = os.path.join(root, fname)
                    targets.append(path)
    if class_mode is None:
        return inputs
    else:
        return inputs, targets


class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def one_epoch(self):
        """Return an iterator that will loop through all the samples one time"""
        return DatasetIter(self)

    def __iter__(self):
        """Return an iterator that will loop through all the samples one time"""
        return DatasetIter(self)

    def __next__(self):
        """Return the next batch in the data. If this batch is the last
        batch in the data, the iterator will be reset -- allowing you
        to loop through the data ad infinitum
        """
        new_batch = next(self._iter)
        self.batches_seen += 1
        if self.batches_seen % self.nb_batches == 0:
            #print('Last Batch of Current Epoch')
            self._iter = DatasetIter(self)
        return new_batch

    next = __next__


class FolderDataset(Dataset):

    def __init__(self, 
                 root,
                 class_mode='label',
                 input_regex='*',
                 target_regex=None,
                 transform=None, 
                 target_transform=None,
                 co_transform=None, 
                 loader='npy',
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 num_workers=0,
                 collate_fn=default_collate, 
                 pin_memory=False):
        """Dataset class for loading out-of-memory data.

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

        loader : string in `{'npy', 'pil', 'nifti'} or function
            defines how to load samples from file
            if a function is provided, it should take in a file path
            as input and return the loaded sample.

        Examples
        --------
        For loading input images and target images (e.g. image and its segmentation):
            >>> data = FolderDataset(root=/path/to/main/dir,
                    class_mode='image', input_regex='*input*',
                    target_regex='*segment*', loader='pil')

        For loading input images with sub-directory as class label:
            >>> data = FolderDataset(root=/path/to/main/dir,
                    class_mode='label', loader='pil')
        """
        if loader == 'npy':
            loader = npy_loader
        elif loader == 'pil':
            loader = pil_loader
        elif loader == 'nifti':
            loader = nifti_loader

        root = os.path.expanduser(root)

        classes, class_to_idx = find_classes(root)
        inputs, targets = make_dataset(root, class_mode,
            class_to_idx, input_regex, target_regex)

        if len(inputs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = os.path.expanduser(root)
        self.inputs = inputs
        self.targets = targets
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader
        self.class_mode = class_mode

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(nb_samples=len(self.inputs))
        elif not shuffle:
            self.sampler = SequentialSampler(nb_samples=len(self.inputs))

        if class_mode == 'image':
            print('Found %i input images and %i target images' %
                (len(self.inputs), len(self.targets)))
        elif class_mode == 'label':
            print('Found %i input images across %i classes' %
                (len(self.inputs), len(self.classes)))  

        self.batches_seen = 0
        self.nb_batches = int(math.ceil(len(self.sampler) / float(self.batch_size)))
        self._iter = DatasetIter(self)      

    def __getitem__(self, index):
        # get paths
        input_sample = self.inputs[index]
        target_sample = self.targets[index]

        # load samples into memory
        input_sample = self.loader(os.path.join(self.root, input_sample))
        if self.class_mode == 'image':
            target_sample = self.loader(os.path.join(self.root, target_sample))
        
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


class TensorDataset(Dataset):

    def __init__(self,
                 input_tensor,
                 target_tensor=None,
                 transform=None, 
                 target_transform=None,
                 co_transform=None,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 num_workers=0,
                 collate_fn=default_collate, 
                 pin_memory=False):
        """Dataset class for loading in-memory data.

        Arguments
        ---------
        input_tensor : torch tensor

        target_tensor : torch tensor

        transform : torch transform
            transform to apply to input sample individually

        target_transform : torch transform
            transform to apply to target sample individually

        loader : string in `{'npy', 'pil', 'nifti'} or function
            defines how to load samples from file
            if a function is provided, it should take in a file path
            as input and return the loaded sample.

        Examples
        --------
        For loading input images and target images (e.g. image and its segmentation):
            >>> data = FolderDataset(root=/path/to/main/dir,
                    class_mode='image', input_regex='*input*',
                    target_regex='*segment*', loader='pil')

        For loading input images with sub-directory as class label:
            >>> data = FolderDataset(root=/path/to/main/dir,
                    class_mode='label', loader='pil')
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

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory

        if sampler is not None:
            self.sampler = sampler
        else:
            if shuffle:
                self.sampler = RandomSampler(nb_samples=len(self.inputs))
            elif not shuffle:
                self.sampler = SequentialSampler(nb_samples=len(self.inputs))

        self.batches_seen = 0
        self.nb_batches = int(math.ceil(len(self.sampler) / float(self.batch_size)))
        self._iter = DatasetIter(self)      

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


