"""
Classify CIFAR10 using a ConvNet and the TensorDataset sampler
"""

import numpy as np
from torchvision.datasets import CIFAR10

data = CIFAR10(root='/users/ncullen/desktop/DATA/cifar/', train=True, 
    download=True)
x_train = data.train_data # numpy array (50k, 3, 32, 32)
y_train = np.array(data.train_labels) # numpy array (50k,)
x_train = x_train.astype('float32') / 255.

data = CIFAR10(root='/users/ncullen/desktop/DATA/cifar/', train=False, 
    download=True)
x_test = data.test_data # (10k, 3, 32, 32)
y_test = np.array(data.test_labels) # (10k,)

import torch
# convert these to torch tensors
x_train = torch.from_numpy(x_train.astype('float32')).float()
y_train = torch.from_numpy(y_train.astype('uint8')).long()
x_test = torch.from_numpy(x_test.astype('float32')).float()
y_test = torch.from_numpy(y_test.astype('uint8')).long()

## Build the network
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchsample import TensorDataset

class Network(nn.Module):
    
    def __init__(self, input_size=(3,32,32)):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.Conv2d(32,32,3),
            nn.ReLU(),
            nn.MaxPool2d((3,3))
        )
        self.flat_fts = self.get_flat_fts(input_size, self.features)
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_fts, 100),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.LogSoftmax()
        )
    
    def get_flat_fts(self, in_size, fts):
        f = fts(Variable(torch.ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))
    
    def forward(self, x):
        fts = self.features(x)
        flat_fts = fts.view(-1, self.flat_fts)
        out = self.classifier(flat_fts)
        return out

    def set_optimizer(self, opt):
        self.opt = opt

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def fit(self, x, y, batch_size, nb_epoch):
        train_loader = TensorDataset(x, y, batch_size=batch_size)
        for epoch in range(nb_epoch):
            self.train_loop(train_loader)

    def fit_loader(self, loader, nb_epoch):
        for epoch in range(nb_epoch):
            self.train_loop(loader)

    def train_loop(self, loader):
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            loss = self.batch_loop(x_batch, y_batch)
            if batch_idx % 10 == 0:
                print('Batch {} - Loss : {:.02f}'.format(batch_idx, 
                    loss))

    def batch_loop(self, x_batch, y_batch):
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)
        self.opt.zero_grad()
        ypred = self(x_batch)
        loss = self.loss_fn(ypred, y_batch)
        loss.backward()
        self.opt.step()

        return loss.data[0]

## Test that the network actually produces the correctly-sized output
net = Network()
net.set_optimizer(optim.Adam(net.parameters()))
net.set_loss(F.nll_loss)

NB_EPOCH = 10
BATCH_SIZE = 32

net.fit(x_train, y_train, batch_size=32, nb_epoch=10)


