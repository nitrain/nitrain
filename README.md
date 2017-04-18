# torch-sample : data augmentation and sampling for pytorch


![Original](https://github.com/ncullen93/torchsample/blob/master/tutorials/imgs/orig1.png "Original") ![Transformed](https://github.com/ncullen93/torchsample/blob/master/tutorials/imgs/tform1.png "Transformed")

This package provides a set of transforms and data structures for sampling from in-memory or out-of-memory data. 
I'm actively  taking requests for new transforms or new features to the samplers. 
(see [for example](https://github.com/ncullen93/torchsample/issues/1))

## Unique Features
- affine transforms which depend only on pytorch (not numpy)
- transforms directly on arbitrary torch tensors (rather than just PIL images)
- perform transforms on data where both input and target are images
- sample and transform images with no target
- save transformed/augmented images to file
- sample arbitrary data directly from folders with speed
- stratified sampling
- variable batch size (e.g. `loader.next_batch(10)`)
- sample for a fixed number of batches without using an `epoch` loop

## Utility Functions
- `torch.pearsonr` (mimics scipy.stats.pearsonr)
- `torch.corrcoef` (mimics np.corrcoef)
- `torch.meshgrid` (mimics np.meshgrid)
- `torch.affine_transform` (functional affine transform)

## Tutorials and examples
- [torchsample overview](https://github.com/ncullen93/torchsample/blob/master/tutorials/torchsample%20tutorial.ipynb) 
- [samplers overview](https://github.com/ncullen93/torchsample/blob/master/tutorials/Samplers.ipynb)
- [cifar in-memory](https://github.com/ncullen93/torchsample/blob/master/examples/cifar_TensorDataset.py)
- [stratified sampling](https://github.com/ncullen93/torchsample/blob/master/examples/stratified_sampling.py)
- [variable batch size](https://github.com/ncullen93/torchsample/blob/master/examples/variable_batchsize.py)

## Example
Perform transforms on datasets where both inputs and targets are images in-memory:

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

Load data out-of-memory using regular expressions:

```python
from torchsample import FolderDataset
train_loader = FolderDataset(root='/users/ncullen/desktop/my_data/', 
    input_regex='*img*', target_regex='*mask*',
    batch_size=32, co_transform=tform, batch_size=3)

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
