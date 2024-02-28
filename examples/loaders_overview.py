# An example of how to download a dataset

import os
import numpy as np
import ants
import torch
import keras
from keras import layers
from nitrain import utils, datasets, loaders, transforms as tx

download = utils.fetch_data('openneuro/ds004711')
def transform_fn(img):
    img = img.resample_image((4,4,4))
    img = img.slice_image(2, 32)
    img = img.mask_image(ants.get_mask(img))
    return img

# use a Datalad/BIDS folder -> create a FolderDataset 
ds = datasets.FolderDataset(path = download.path, 
                        layout = 'bids',
                        x_config = {'suffix': 'T1w', 'run': [None, '01']},
                        y_config = {'column': 'age'},
                        x_transform = transform_fn)
# load into memory + create memory dataset to make things faster
x, y = ds[:10]
y = np.arange(len(y))

ds2 = datasets.MemoryDataset(x, y)


l = loaders.NumpyLoader(ds2, batch_size=4, 
                        x_transforms=[tx.RandomSlice(axis=2)]) # returns numpy arrays
l2 = loaders.TorchLoader(ds2, batch_size=4) # returns torch tensors
#l3 = loaders.TFLoader(ds2, batch_size=4) # returns tf tensors
l3 = loaders.KerasLoader(ds2, batch_size=4)

for a, b in l:
    print(b)

for a, b in l2:
    print(b)

            
# build simple keras model
inputs = keras.Input(shape=(64,64,1))
x = keras.layers.Conv2D(8, 3, strides=2)(inputs)
x = keras.layers.MaxPooling2D(pool_size=2)(x)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(1, activation=None)(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanSquaredError(name="mse")],
)

model.fit(l, epochs=100)

# predict on train dataset
x_test, y_test = ds[:]
y_pred = model.predict(np.array(x_test)).flatten()

# plot true age versus predicted age
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.show()