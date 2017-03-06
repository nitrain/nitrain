# torch-sample : data augmentation and sampling for pytorch


![Original](https://github.com/ncullen93/torchsample/blob/master/tutorials/imgs/orig1.png "Original") ![Transformed](https://github.com/ncullen93/torchsample/blob/master/tutorials/imgs/tform1.png "Transformed")

This package provides a set of transforms and data structures for sampling from in-memory or out-of-memory data. 
I'm actively  taking requests for new transforms or new features to the samplers. 
(see [for example](https://github.com/ncullen93/torchsample/issues/1))

## UNIQUE FEATURES
- affine transforms
- transforms directly on arbitrary torch tensors (rather than just PIL images)
- perform transforms on data where both input and target are images
- sample arbitrary data directly from folders with speed
- stratified sampling
- variable batch size (e.g. `loader.next_batch(10)`)
- sample for a fixed number of batches without using an `epoch` loop

## Tutorials


## Example
Perform transforms on datasets where both inputs and targets are images:

```python
from torchvision.datasets import MNIST
train = MNIST(root='/users/ncullen/desktop/data/', train=True, download=True)
x_train = train.train_data

process = Compose([TypeCast('float'), AddChannel(), RangeNormalize(0,1)])
affine = Affine(rotation_range=30, zoom_range=(1.0,1.4), shear_range=0.1,
    translation_range=(0.2,0.2))
tform = Compose([process, affine])
train_loader = TensorDataset(x_train, x_train, co_transform=tform, batch_size=3)

x_batch, y_batch = train_loader.next_batch()

```

## Transforms

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
The following transforms perform affine (or affine-like) transforms on torch tensors. 

- `Rotate()` 
- `Translate()` 
- `Shear()` 
- `Zoom()` 

We also provide a class for stringing multiple affine transformations together so that only one interpolation takes place:

- `Affine()` 
- `AffineCompose()` 

## Sampling
We provide the following datasets which provide general structure and iterators for sampling from and using transforms on in-memory or out-of-memory data:

- `TensorDataset()` 

- `FolderDataset()` 

### Sampling Features
- Stratified Sampling
- sample a fixed number of batches without an `epoch` loop
- sample/augmentation without any target tensor
- use a regular expression to find or filter out certain images
- Sample datasets with both input and target images
- Apply the same augmentation/affine transforms to input and target images
- save transformed/augmented images to file
