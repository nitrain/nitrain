# An example of how to download a dataset

import os
import bids
import nibabel
import datalad.api as dl
import numpy as np
import pandas as pd
from nitrain import utils, data, transforms as tx

download = utils.fetch_datalad('ds004711')

# use a Datalad/BIDS folder -> create a FolderDataset 
ds = data.FolderDataset(path = download.path, 
                        layout = 'bids',
                        x_config = {'suffix': 'T1w', 
                                    'scope': 'derivatives',
                                    'desc': 'precompute'},
                        y_config = {'column': 'age'})
# load in some data
x, y = ds[:]

## create a memory dataset for 
ds_memory = data.MemoryDataset(x, y)

loader = data.DatasetLoader(ds_memory,
                            batch_size=10)

# loop through each x, y pair for one epoch
for x, y in loader:
    print(x.shape)
    print(y)

# build simple keras model
import keras
from keras import layers

inputs = keras.Input(shape=(64,64,1))
x = layers.Conv2D(8, 3, strides=2)(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation=None)(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.MeanSquaredError(name="mse")],
)

for epoch in range(10):
    print(epoch)
    for x_batch, y_batch in loader:
        model.train_on_batch(x_batch, y_batch)
        print(model.test_on_batch(x_batch, y_batch)[1])


loader = data.DatasetLoader(ds_memory,
                            batch_size=150)
x_test, y_test = next(iter(loader))
y_pred = model.predict(x_test).flatten()

# plot true age versus predicted age
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.show()