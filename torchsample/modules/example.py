

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

# constraints
# -> Nonneg on Conv layers applied at end of every epoch
# -> UnitNorm on FC layers applied every 4 batches
from torchsample.constraints import NonNeg, UnitNorm
constraints = [NonNeg(frequency=1, unit='batch', module_filter='*conv*'),
               UnitNorm(frequency=4, unit='batch', module_filter='*fc*')]

# regularizers 
# -> L1 on Conv layers
# -> L2 on FC layers
from torchsample.regularizers import L1Regularizer, L2Regularizer
regularizers = [L1Regularizer(scale=1e-6, module_filter='*conv*'),
                L2Regularizer(scale=1e-6, module_filter='*fc*')]

# callbacks
# lambda callback
from torchsample.callbacks import LambdaCallback
callbacks = [LambdaCallback(on_train_end=lambda logs: print('TRAINING FINISHED'))]

model = Network()
model.set_loss(F.nll_loss)
model.set_optimizer(optim.Adadelta, lr=1.0)
model.set_regularizers(regularizers)
model.set_constraints(constraints)
model.set_callbacks(callbacks)

# FIT THE MODEL
model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          nb_epoch=5, 
          batch_size=128,
          verbose=1)

# SAVE MODEL PARAMETERS (doesnt save architecture)
model.save_state_dict('/users/ncullen/desktop/mymodel.t7')

# EVALUATE ON TEST DATA
val_loss = model.evaluate(x_test, y_test)
print('Function Val Loss: ' , val_loss)

# PREDICT TEST DATA (then manually evaluate)
from torch.autograd import Variable
y_pred = model.predict(x_test)
val_loss = model._loss(y_pred, Variable(y_test.long()))
print('Manual Val Loss: ', val_loss.data[0])

# CREATE A NEW MODEL AND EVALUATE WITHOUT LOADING/TRAINING
new_model = Network()
new_model.set_loss(F.nll_loss)
new_model.set_optimizer(optim.Adadelta, lr=1.0)
#model.set_constraints(constraints)
new_model.set_regularizers(regularizers)
# evaluation loop
val_loss = new_model.evaluate(x_test, y_test)
print('Re-initialized Model Val Loss (should be large): ' , val_loss)

# LOAD PREVIOUS MODEL PARAMS (then evaluate to show it works)
new_model.load_state_dict(torch.load('/users/ncullen/desktop/mymodel.t7'))
# evaluation loop
val_loss = new_model.evaluate(x_test, y_test)
print('File-Loaded Model Val Loss: ' , val_loss)




