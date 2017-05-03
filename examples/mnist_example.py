
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torchsample.modules import ModuleTrainer
from torchsample.callbacks import *
from torchsample.regularizers import *
from torchsample.constraints import *
from torchsample.initializers import *
from torchsample.metrics import *

import os
from torchvision import datasets
ROOT = '/users/ncullen/data'
dataset = datasets.MNIST(ROOT, train=True, download=True)
x_train, y_train = torch.load(os.path.join(dataset.root, 'processed/training.pt'))
x_test, y_test = torch.load(os.path.join(dataset.root, 'processed/test.pt'))

x_train = x_train.float()
y_train = y_train.long()
x_test = x_test.float()
y_test = y_test.long()

x_train = x_train / 255.
x_test = x_test / 255.
x_train = x_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)

# only train on a subset
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]


# Define your model EXACTLY as if you were using nn.Module
class Network(nn.Module):
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
trainer = ModuleTrainer(model)

callbacks = [EarlyStopping(patience=10),
             ReduceLROnPlateau(factor=0.5, patience=5)]
regularizers = [L1Regularizer(scale=1e-3, module_filter='conv*'),
                L2Regularizer(scale=1e-5, module_filter='fc*')]
constraints = [UnitNorm(frequency=3, unit='batch', module_filter='fc*'),
               MaxNorm(value=2., lagrangian=True, scale=1e-2, module_filter='conv*')]
initializers = [XavierUniform(bias=False, module_filter='fc*')]
metrics = [CategoricalAccuracy(top_k=3)]

trainer.compile(loss='nll_loss',
                optimizer='adadelta')
                #regularizers=regularizers,
                #constraints=constraints,
                #initializers=initializers,
                #metrics=metrics)

trainer.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          nb_epoch=20, 
          batch_size=128,
          verbose=1)