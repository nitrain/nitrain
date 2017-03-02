import torch
import numpy as np
import math

class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class MultiSampler(Sampler):
    """Samples elements more than once in a single pass through the data.

    This allows the number of samples per epoch to be larger than the number
    of samples itself, which can be useful when training on 2D slices taken
    from 3D images, for instance.
    """
    def __init__(self, nb_samples, desired_samples, shuffle=False):
        """Initialize MultiSampler

        Arguments
        ---------
        data_source : the dataset to sample from
        
        desired_samples : number of samples per batch you want
            whatever the difference is between an even division will
            be randomly selected from the samples.
            e.g. if len(data_source) = 3 and desired_samples = 4, then
            all 3 samples will be included and the last sample will be
            randomly chosen from the 3 original samples.

        shuffle : boolean
            whether to shuffle the indices or not
        """
        data_samples = nb_samples

        n_repeats = desired_samples / data_samples

        cat_list = []
        for i in range(math.floor(n_repeats)):
            cat_list.append(np.arange(data_samples))
        # add the left over samples
        left_over = desired_samples % data_samples
        cat_list.append(np.random.choice(data_samples,left_over))
        self.sample_idx_array = np.concatenate(tuple(cat_list))
        if shuffle:
            self.sample_idx_array = np.random.permutation(self.sample_idx_array)

    def __iter__(self):
        return iter(self.sample_idx_array)

    def __len__(self):
        return len(self.sample_idx_array)


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, nb_samples):
        self.num_samples = nb_samples

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples


class RandomSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, nb_samples):
        self.num_samples = nb_samples

    def __iter__(self):
        return iter(torch.randperm(self.num_samples).long())

    def __len__(self):
        return self.num_samples
