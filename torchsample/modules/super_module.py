"""
SuperModule for high level training on Pytorch models
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import TensorDataset, DataLoader

class SuperModule(nn.Module):

    def __init__(self):
        """
        Properties:
        - optimizers
        - losses
        - callbacks
        - regularizers
        - normalizers
        """
        super(SuperModule, self).__init__()
        self.opts = []
        self.losses = []

        self.callbacks = None
        self.regularizers = None
        self.normalizers = None

    def forward(self, *input):
        """Defines the computation performed at every call.

        Should be overriden by all subclasses.
        """
        raise NotImplementedError('Subclass must implement this method')

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer(self, optimizer, lr=1e-4):
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def fit(self, x, y, nb_epoch=100, batch_size=32, verbose=1):
        train_dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.fit_loader(loader=train_loader, nb_epoch=nb_epoch, verbose=verbose)

    def fit_loader(self, loader, nb_epoch=100, verbose=1):
        for epoch in range(nb_epoch):
            for batch_idx, (x_batch, y_batch) in enumerate(loader):
                # Convert numpy array to torch Variable
                inputs = Variable(x_batch)
                targets = Variable(y_batch)

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()
                outputs = self(inputs)
                # compute model loss
                loss = self.loss(outputs, targets)

                # add regularizer loss
                for param in self.parameters():
                    # filter out bias
                    if param.dim() > 1:
                        loss += self.reg_loss(param)

                loss.backward()
                self.optimizer.step()
                
                if (epoch+1) % 5 == 0:
                    print ('Epoch [%d/%d], Loss: %.4f' 
                           %(epoch+1, nb_epoch, loss.data[0]))

    def predict(self, x):
        x_var = Variable(torch.from_numpy(x))
        y_pred = self(x_var)
        return y_pred.data.numpy()

    def evaluate(self, x, y):
        x_var = Variable(torch.from_numpy(x))
        y_var = Variable(torch.from_numpy(y))

        y_pred = self(x_var)
        loss = self.loss(y_pred, y_var)
        return loss.data.numpy()[0]











