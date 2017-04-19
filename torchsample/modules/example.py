
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

x_train = torch.from_numpy(x_train[:10000])
y_train = torch.from_numpy(y_train[:10000])
x_test = torch.from_numpy(x_test[:1000])
y_test = torch.from_numpy(y_test[:1000])

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

# constraints
# -> Nonneg on Conv layers applied at end of every epoch
# -> UnitNorm on FC layers applied every 4 batches
#from torchsample.modules import NonNeg, UnitNorm
#constraints = [NonNeg(frequency=1, unit='batch', module_filter='*conv*'),
#               UnitNorm(frequency=4, unit='batch', module_filter='*fc*')]

# regularizers -> L1 on Conv layers, L2 on FC layers
from torchsample.modules import L1Regularizer, L2Regularizer
regularizers = [L1Regularizer(scale=1e-6, module_filter='*conv*'),
                L2Regularizer(scale=1e-6, module_filter='*fc*')]

model = Network()
model.set_loss(F.nll_loss)
model.set_optimizer(optim.Adadelta, lr=1.0)
#model.set_constraints(constraints)
model.set_regularizers(regularizers)

import time
s = time.time()
model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          nb_epoch=5, 
          batch_size=128,
          verbose=1)
e = time.time()
#print('\n\nTrain Time:' , round(e-s,4), 'seconds')

# evaluation loop
val_loss = model.evaluate(x_test, y_test)
print('Val Loss: ' , val_loss)

# prediction and manual evaluation
from torch.autograd import Variable
y_pred = model.predict(x_test)
val_loss = model._loss(Variable(y_pred), Variable(y_test.long()))
print('Manual Val Loss: ', val_loss.data[0])


