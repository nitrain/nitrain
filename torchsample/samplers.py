
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

class StratifiedSampler(Sampler):
    """Stratified Sampling

    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        from sklearn.model_selection import StratifiedShuffleSplit
        s = StratifiedShuffleSplit(n_splits=self.n_splits, 
            test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

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
        self.data_samples = nb_samples
        self.desired_samples = desired_samples
        self.shuffle = shuffle

    def gen_sample_array(self):
        n_repeats = self.desired_samples / self.data_samples
        cat_list = []
        for i in range(math.floor(n_repeats)):
            cat_list.append(np.arange(self.data_samples))
        # add the left over samples
        left_over = self.desired_samples % self.data_samples
        cat_list.append(np.random.choice(self.data_samples, left_over))
        self.sample_idx_array = np.concatenate(tuple(cat_list))
        if self.shuffle:
            self.sample_idx_array = np.random.permutation(self.sample_idx_array)
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return self.desired_samples


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
