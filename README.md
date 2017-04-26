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

Torchsample provides a wide range of <b>callbacks</b>, generally mimicking the interface
found in `Keras`:

- `EarlyStopping(file, 
                 monitor='val_loss', 
                 save_best_only=False, 
                 save_weights_only=True,
                 max_checkpoints=-1,
                 verbose=0)`
- `ModelCheckpoint(monitor='val_loss',
                   min_delta=0,
                   patience=0)`
- `LearningRateScheduler(schedule)`
- `ReduceLROnPlateau(monitor='val_loss', 
                     factor=0.1, 
                     patience=10,
                     epsilon=0, 
                     cooldown=0, 
                     min_lr=0,
                     verbose=0)`
- `CSVLogger(file, 
             separator=',', 
             append=False)`


```python
from torchsample.callbacks import EarlyStopping

callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
model.set_callbacks(callbacks)
```

Torchsample also provides <b>regularizers</b>:

- `L1Regularizer(scale=0.0, module_filter='*')`
- `L2Regularizer(scale=0.0, module_filter='*')`
- `L1L2Regularizer(l1_scale=0.0, l2_scale=0.0, module_filter='*')`


and <b>constraints</b>:

```python
UnitNorm(frequency=1, 
         unit='batch',
         lagrangian=False,
         scale=0.,
         module_filter='*')
```

```python
MaxNorm(value, 
        axis=1, 
        frequency=1, 
        unit='batch',
        lagrangian=False,
        scale=0.,
        module_filter='*')
```
```python
NonNeg(frequency=1, 
       unit='batch',
       lagrangian=False,
       scale=0.,
       module_filter='*')
```

Both regularizers and constraints can be selectively applied on layers using regular expressions and the `module_filter`
argument. Constraints can be explicit (hard) constraints applied at an arbitrary batch or
epoch frequency, or they can be implicit (soft) constraints similar to regularizers
where the the constraint deviation is added as a penalty to the total model loss.

```python
from torchsample.constraints import MaxNorm, NonNeg
from torchsample.regularizers import L1Regularizer

# hard constraint applied every 5 batches
hard_constraint = MaxNorm(value=2., frequency=5, unit='batch', module_filter='*fc*')
# implicit constraint added as a penalty term to model loss
soft_constraint = NonNeg(lagrangian=True, scale=1e-3, module_filter='*fc*')
constraints = [hard_constraint, soft_constraint]
model.set_constraints(constraints)

regularizers = [L1Regularizer(scale=1e-4, module_filter='*conv*')]
model.set_regularizers(regularizers)
```

You can also fit directly on a `torch.utils.data.DataLoader` and can have
a validation set as well :

```python
from torchsample import TensorDataset
from torch.utils.data import DataLoader

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32)

val_dataset = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32)

model.fit_loader(loader, val_loader=val_loader, nb_epoch=100)
```

## Utility Functions
Finally, torchsample provides a few utility functions not commonly found:

### Tensor Functions
- `th_iterproduct` (mimics itertools.product)
- `th_gather_nd` (N-dimensional version of torch.gather)
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

