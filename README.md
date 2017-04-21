# torchsample: Data Augmentation, Loading, and High-Level Modules

This package provides a few things;
- Comprehensive data augmentation, sampling, and loading
- A high-level module for Keras-like training with callbacks, constraints, and regularizers.
- Utility functions not commonly found elsewhere

## `SuperModule`
The `SuperModule` class provides a high-level training interface which abstracts
away the training loop while providing callbacks, constraints, and regularizers. 
Most importantly, you lose ZERO flexibility since this model inherits directly
from `nn.Module`.


## Data Augmentation and Datasets
The torchsample package provides a ton of good data augmentation and transformation
tools which can be applied during data loading. The package also provides the flexible
`TensorDataset` and `FolderDataset` classes to handle most dataset needs.

### Torch Transforms
These transforms work directly on torch tensors

- `Compose()` 
- `AddChannel()`
- `SwapDims()` 
- `RangeNormalize()` 
- `StdNormalize()` 
- `Slice2D()` 
- `RandomCrop()` 
- `SpecialCrop()` 
- `Pad()` 
- `RandomFlip()` 
- `ToTensor()` 

### Affine Transforms
![Original](https://github.com/ncullen93/torchsample/blob/master/tutorials/imgs/orig1.png "Original") ![Transformed](https://github.com/ncullen93/torchsample/blob/master/tutorials/imgs/tform1.png "Transformed")

The following transforms perform affine (or affine-like) transforms on torch tensors. 

- `Rotate()` 
- `Translate()` 
- `Shear()` 
- `Zoom()` 

We also provide a class for stringing multiple affine transformations together so that only one interpolation takes place:

- `Affine()` 
- `AffineCompose()` 

### Sampling
We provide the following datasets which provide general structure and iterators for sampling from and using transforms on in-memory or out-of-memory data:

- `TensorDataset()` 

- `FolderDataset()` 


## Utility Functions
Finally, torchsample provides a few utility functions not commonly found:

- `th_meshgrid` (mimics np.meshgrid)
- `th_random_choice` (mimics np.random.choice)
- `th_pearsonr` (mimics scipy.stats.pearsonr)
- `th_corrcoef` (mimics np.corrcoef)
- `th_affine_transform` (functional affine transform)
