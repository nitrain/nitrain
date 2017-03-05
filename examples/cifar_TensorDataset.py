"""
Classify CIFAR10 using a ConvNet and the TensorDataset sampler
"""

import numpy as np
from torchvision.datasets import CIFAR10

data = CIFAR10(root='/users/ncullen/desktop/DATA/cifar/', train=True, 
    download=True)
x_train = data.train_data # numpy array (50k, 3, 32, 32)
y_train = np.array(data.train_labels) # numpy array (50k,)

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


## Test that the network actually produces the correctly-sized output
net = Network()

## create optimizer
opt = optim.Adam(net.parameters())

## create sampler
from torchsample import TensorDataset
from torchsample.transforms import RangeNormalize

NB_EPOCH = 10
BATCH_SIZE = 32

train_sampler = TensorDataset(x_train, y_train,
    transform=RangeNormalize(0.,1.,n_channels=3),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

for epoch in range(NB_EPOCH):

    for batch_idx, (x_batch, y_batch) in enumerate(train_sampler):
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)
        opt.zero_grad()
        ypred = net(x_batch)
        loss = F.nll_loss(ypred, y_batch)
        loss.backward()
        opt.step()

        if batch_idx % 10 == 0:
            print('Batch {} / {} - Loss : {:.02f}'.format(batch_idx, 
                train_sampler.nb_batches, loss.data[0]))







