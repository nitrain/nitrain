

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## LOAD DATA
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
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]


## Create your model exactly as you would with `nn.Module`
from torchsample.modules import SuperModule
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

# callbacks
# lambda callback
#from torchsample.callbacks import ModelCheckpoint
#callbacks = [ModelCheckpoint(file='/users/ncullen/desktop/test/model_{epoch}_{loss}.pt',
#                             monitor='val_loss',
#                             save_best_only=False,
#                             max_checkpoints=3)]
#from torchsample.callbacks import CSVLogger
#callbacks = [CSVLogger(file='/users/ncullen/desktop/test/logger.csv',append=True)]
#from torchsample.callbacks import EarlyStopping
#callbacks = [EarlyStopping(monitor='val_loss',
#                           min_delta=0,
#                           patience=2)]
#from torchsample.callbacks import LearningRateScheduler
#save_lrs = []
#def lr_schedule(epoch, lr, **kwargs):
#    """exponential decay"""
#    new_lr = lr[0] * 1e-5**(epoch / 200)
#    save_lrs.append(new_lr)
#    return new_lr
#callbacks = [LearningRateScheduler(lr_schedule)]
#import matplotlib.pyplot as plt
#%matplotlib inline
#plt.plot(np.arange(len(save_lrs)), np.array(save_lrs))
#plt.show()
from torchsample.callbacks import ReduceLROnPlateau
callbacks = [ReduceLROnPlateau(monitor='val_loss', 
                               factor=0.1, 
                               patience=1,
                               cooldown=0, 
                               min_lr=1e-3,
                               verbose=1)]
model = Network()
model.set_loss(F.nll_loss)
model.set_optimizer(optim.Adadelta, lr=1.0)
model.set_callbacks(callbacks)

# FIT THE MODEL
model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          nb_epoch=20, 
          batch_size=128,
          verbose=1)









