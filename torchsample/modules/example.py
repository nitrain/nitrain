
from torchsample.modules import SuperModule

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
x_train = np.expand_dims(x_train,1).astype('float32')
x_test = np.expand_dims(x_test,1).astype('float32')


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
model.set_optimizer(optim.Adadelta)

import time
s = time.time()
model.fit(torch.from_numpy(x_train[:10000]), 
          torch.from_numpy(y_train[:10000]), 
          val_data=(torch.from_numpy(x_test), torch.from_numpy(y_test)),
          nb_epoch=2, 
          batch_size=128)
e = time.time()






