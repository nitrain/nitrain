# pytorch-sample : data augmentation for pytorch

This package provides a set of transforms and data structures for sampling from in-memory or out-of-memory data. I'm openly taking requests for new transforms or new features to the samplers. 

## Transforms

### Torch Transforms
These transforms work directly on torch tensors

- `Compose()` : string together multiple transforms
- `RangeNormalize()` : normalize a tensor between a `min` and `max` value (e.g. between (-1,1), (0,1), etc.)
- `StdNormalize()` : normalize a tensor to have zero mean and unit variance
- `Slice2D()` : take a random 2D slice from a 3D image (or video) along a given axis
- `RandomCrop()` : take a random crop of a given size from a 2D images
- `SpecialCrop()` : take a crop of a given size from one of the four corners of an image or the center of an image
- `Pad()` : pad an image by a given size
- `Flip()` : randomly flip a given image horizontally and/or vertical with a given probability
- `ToTensor()` : convert numpy array or pil image to torch tensor

### Affine Transforms
The following transforms perform affine (or affine-like) transforms on torch tensors. 

- `Rotation()` : randomly rotate an image between given degree bounds
- `Translation()` : randomly translate an image horizontally and/or vertically between given bounds
- `Shear()` : randomly shear an image between given radian bounds
- `Zoom()` : randomly zoom in or out on an image between given percentage bounds

We also provide a class for stringing multiple affine transformations together so that only one interpolation takes place:

- `Affine()` : perform an affine transform with all of the above options available as arguments to this function, with the benefit of using only one interpolation

- `AffineCompose()` : perform a string of explicitly-provided affine transforms, with the benefit of using only one interpolation.

## Sampling
We provide the following datasets which provide general structure and iterators for sampling from and using transforms on in-memory or out-of-memory data:

- `TensorDataset()` : sample from and/or iterate through an input and target tensor, while providing transforms and a sampling procedure.

- `FolderDataset()` : sample from and/or iterate images or arbitrary data types existing in directories, which will only be loaded into memory as needed.

### Sampling Features
- use a regular expression to find or filter out certain images
- Load input and target images from the same folder and identify them using regular expressions
- Apply the same augmentation/affine transforms to input and target images

## Examples & Tutorial

### TensorDataset Examples
The `TensorDatset` provides a class structure for sampling from data that is already loaded into memory as torch tensors.

Here is the class signature:

```python
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
                 pin_memory=False)
```

#### `TensorDataset` Explained
You can see the most obvious argument - `input_tensor`, which would be your input images/data. This can be of arbitrary size, but the first dimension should always be the number of samples and if the input represents an image then the channel dimension should be before the other dimensions (e.g. `(Channels, Height, Width)`).

There is also an <b>optional</b> target tensor, which might be a vector of integer classifications, continous values, or even another set of images or arbitrary data.

Next, there is the `transform`, which takes in the input tensor, performs some operation on it, and returns the modified version. The `target_transform` does the same thing with the `target_tensor`. 

There is also a `co_transform` argument to perform transforms on the `input_tensor` and `target_tensor` together. This is particularly useful for segmentation tasks, where we may want to perform the same affine transform on both the input image and the segmented image. Note that the `co_transform` will occur <b>after</b> the individual transforms.

There is a `batch_size` argument, which determines how many samples to take for each batch. There is also the boolean `shuffle` argument, which determines whether we will take samples sequentially as given or in a random order. 

Finally, there are a few arguments which relate to the multi-processing nature of the samping, which I wont get into but will refer interested readers to the official pytorch docs.



