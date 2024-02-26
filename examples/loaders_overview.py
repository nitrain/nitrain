# An example of how to download a dataset

import os
import numpy as np
import ants
import torch
import keras
from keras import layers
from nitrain import utils, data, transforms as tx

download = utils.fetch_datalad('ds004711')
ds_pre = data.FolderDataset(path = download.path, 
                            layout = 'bids',
                            x_config = {'suffix': 'T1w', 
                                        'scope': 'derivatives', 
                                        'desc': 'precompute'},
                            y_config = {'column': 'age'})
# load into memory + create memory dataset to make things faster
x, y = ds_pre[:]
y = np.arange(len(x))
ds = data.MemoryDataset(x, y, 
                        x_transform=lambda x: np.expand_dims(x.resample_image((4,4)).numpy(),-1))

l = data.DatasetLoader(ds, batch_size=10)
l2 = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True)

# build simple keras model
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

model.fit(l, epochs=20)

# predict on train dataset
x_test, y_test = ds[:]
y_pred = model.predict(np.array(x_test)).flatten()

# plot true age versus predicted age
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.show()