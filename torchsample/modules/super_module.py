"""
SuperModule for high level training on Pytorch models
"""

import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# local imports
from .. import TensorDataset
from . import callbacks as cbks


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
        self.history = cbks.History()

    def forward(self, *input):
        """Defines the computation performed at every call.

        Should be overriden by all subclasses.
        """
        raise NotImplementedError('Subclass must implement this method')

    def set_loss(self, loss):
        self.loss = loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer(self.parameters())

    def fit(self,
            x, 
            y, 
            val_data=None, 
            nb_epoch=100, 
            batch_size=32, 
            callbacks=None, 
            verbose=1):
        """
        Fit a model on torch tensors
        """
        train_dataset = TensorDataset(x, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        if val_data is not None:
            test_dataset = TensorDataset(val_data[0], val_data[1])
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        else:
            test_loader = None
        self.fit_loader(loader=train_loader, test_loader=test_loader,
                        nb_epoch=nb_epoch, callbacks=callbacks, 
                        verbose=verbose)

    def fit_loader(self, 
                   loader, 
                   test_loader=None, 
                   nb_epoch=100, 
                   callbacks=None, 
                   verbose=1):
        """
        Fit a model on a DataLoader
        """
        ## create callbacks
        #self.history = cbks.History()
        #callback_list = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
        #if verbose > 0:
        callback_list = [self.history,cbks.TQDM()]
        callbacks = cbks.CallbackList(callback_list)
        callbacks.set_model(self)
        callbacks.set_params({
            'batch_size': loader.batch_size,
            'nb_epoch': nb_epoch,
            'nb_batches': int(math.ceil(len(loader.dataset.inputs)/loader.batch_size)),
            'verbose': verbose,
        })

        callbacks.on_train_begin()

        for epoch in range(nb_epoch):
            epoch_logs = {
                'epoch_idx': epoch
            }
            callbacks.on_epoch_begin(epoch, epoch_logs)

            for batch_idx,(x_batch, y_batch) in enumerate(loader):

                batch_logs = {
                    'batch_idx': batch_idx,
                }                
                callbacks.on_batch_begin(batch_idx, batch_logs)

                # Convert torch.Tensor to Variable
                inputs = Variable(x_batch)
                targets = Variable(y_batch)

                # zero the gradients
                self.optimizer.zero_grad()
                # make forward pass
                outputs = self(inputs)
                # compute model loss
                loss = self.loss(outputs, targets)
                # make backward pass
                loss.backward()
                # make optimizer step to update weights
                self.optimizer.step()

                batch_logs['loss'] = loss.data[0]
                callbacks.on_batch_end(batch_idx, batch_logs)
            
            if test_loader is not None:
                test_loss = self.evaluate_loader(test_loader)
                epoch_logs['val_loss'] = test_loss

            callbacks.on_epoch_end(epoch, epoch_logs)

        callbacks.on_train_end()

    def predict(self, x):
        x_var = Variable(x)
        y_pred = self(x_var)
        return y_pred.data.numpy()

    def evaluate(self, x, y):
        x_var = Variable(x)
        y_var = Variable(y)

        y_pred = self(x_var)
        loss = self.loss(y_pred, y_var)
        return loss.data.numpy()[0]

    def evaluate_loader(self, loader):
        losses = []
        for x_batch, y_batch in loader:
            x_batch = Variable(x_batch)
            y_batch = Variable(y_batch)

            y_pred = self(x_batch)
            loss = self.loss(y_pred, y_batch)
            losses.append(loss.data[0])

        return torch.mean(torch.FloatTensor(losses))   











