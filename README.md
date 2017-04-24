# High-Level Training, Data Augmentation, and Utilities for Pytorch

This package provides a few things:
- A high-level module for Keras-like training with callbacks, constraints, and regularizers.
- Comprehensive data augmentation, transforms, sampling, and loading
- Utility tensor and variable functions so you don't need numpy as often

## SuperModule
The `SuperModule` class provides a high-level training interface which abstracts
away the training loop while providing callbacks, constraints, and regularizers. 
Most importantly, you lose ZERO flexibility since this model inherits directly
from `nn.Module`.

Example:
```python
from torchsample.modules import SuperModule
# Define your model EXACTLY as if you were using nn.Module
class Network(SuperModule):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Network()
model.set_loss(F.nll_loss)
model.set_optimizer(optim.Adadelta, lr=1.0)

model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          nb_epoch=20, 
          batch_size=128,
          verbose=1)
```

## Utility Functions
Finally, torchsample provides a few utility functions not commonly found:

### Tensor Functions
- `th_meshgrid` (mimics itertools.product and np.meshgrid)

  - ```python
    >>> grid = th_meshgrid(2,2,2)
        #  0     0     0
        #  0     0     1
        #  0     1     0
        #  0     1     1
        #  1     0     0
        #  1     0     1
        #  1     1     0
        #  1     1     1
        # [torch.LongTensor of size 8x3]
    >>> grid = th_meshgrid(2,torch.LongTensor([3,4,5]))
        #  0  3
        #  0  4
        #  0  5
        #  1  3
        #  1  4
        #  1  5
        # [torch.LongTensor of size 6x2]
    >>> grid = th_meshgrid(torch.FloatTensor([3.4, 2.1]), 3, torch.LongTensor([3]))         
        # 3.4000  0.0000  3.0000
        # 3.4000  1.0000  3.0000
        # 3.4000  2.0000  3.0000
        # 2.1000  0.0000  3.0000
        # 2.1000  1.0000  3.0000
        # 2.1000  2.0000  3.0000
        # [torch.FloatTensor of size 6x3]
    ```

- `th_gather_nd` (N-dimensional version of torch.gather)

  - ```python
    >>> x = torch.zeros(2,3,4,5)
    >>> x[0,2,0,1] = 12
    >>> x[1,1,3,1] = 23
    >>> idx = torch.LongTensor([[0,2,0,1],
    ...                         [1,1,3,1]])
    >>> print(th_gather_nd(x,idx))
    #      12
    #      23
    #   [torch.FloatTensor of size 2]
    >>> x = torch.randn(12,5,3)
    >>> coords = th_meshgrid_like(x)
    >>> xx = th_gather_nd(x, coords).view_as(x)
    >>> print(th_allclose(x, xx))
    # True
    ```
- `th_random_choice` (mimics np.random.choice)
- `th_pearsonr` (mimics scipy.stats.pearsonr)
- `th_corrcoef` (mimics np.corrcoef)
- `th_affine2d` and `th_affine3d` (affine transforms on torch.Tensors)

### Variable Functions
- `F_affine2d` and `F_affine3d`
- `F_map_coordinates2d` and `F_map_coordinates3d`

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
![Original](https://github.com/ncullen93/torchsample/blob/master/examples/imgs/orig1.png "Original") ![Transformed](https://github.com/ncullen93/torchsample/blob/master/examples/imgs/tform1.png "Transformed")

The following transforms perform affine (or affine-like) transforms on torch tensors. 

- `Rotate()` 
- `Translate()` 
- `Shear()` 
- `Zoom()` 

We also provide a class for stringing multiple affine transformations together so that only one interpolation takes place:

- `Affine()` 
- `AffineCompose()` 

### Datasets and Sampling
We provide the following datasets which provide general structure and iterators for sampling from and using transforms on in-memory or out-of-memory data:

- `TensorDataset()` 

- `FolderDataset()` 

