"""
SuperModule for high level training on Pytorch models
"""
from __future__ import print_function
from __future__ import absolute_import

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# local imports
from ..datasets import TensorDataset
from ..callbacks import CallbackModule, History, TQDM
from ..constraints import ConstraintModule
from ..regularizers import RegularizerModule


class SuperModule(nn.Module):

    def __init__(self):
        """
        SuperModule for high-level training of Pytorch models

        TODO:
            - allow metrics
                - e.g. for validation accuracy instead of loss
        """
        super(SuperModule, self).__init__()

        self.history = History()
        self._callbacks = [self.history]
        self._constraints = []
        self._regularizers = []
        self.stop_training = False

    def forward(self, *input):
        """
        Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError('Subclass must implement this method')

    def set_loss(self, loss):
        self._loss = loss

    def set_optimizer(self, optimizer, **kwargs):
        if 'parameters' in kwargs:
            parameters = kwargs['parameters']
        else:
            parameters = self.parameters()
        self._optimizer = optimizer(parameters, **kwargs)

    def set_regularizers(self, regularizers):
        self._regularizers = regularizers

    def set_constraints(self, constraints):
        self._constraints = constraints

    def set_callbacks(self, callbacks):
        self._callbacks += callbacks

    def fit(self,
            x, 
            y,
            validation_data=None, 
            nb_epoch=100, 
            batch_size=32, 
            verbose=1):
        """
        Fit a model on torch tensors
        """
        train_dataset = TensorDataset(x, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        if validation_data is not None:
            test_dataset = TensorDataset(validation_data[0], validation_data[1])
            val_loader = DataLoader(test_dataset, batch_size=batch_size)
        else:
            val_loader = None
        self.fit_loader(loader=train_loader, val_loader=val_loader,
                        nb_epoch=nb_epoch, verbose=verbose)

    def fit_on_batch(self, x, y):
        inputs = Variable(x)
        targets = Variable(y)

        # zero the gradients
        self._optimizer.zero_grad()
        # make forward pass
        outputs = self(inputs)
        # compute model loss
        loss = self._loss(outputs, targets)
        reg_loss = self._regularizers.compute_loss()
        total_loss = loss + reg_loss
        # make backward pass
        total_loss.backward()
        # make optimizer step to update weights
        self._optimizer.step()

    def fit_loader(self, 
                   loader, 
                   val_loader=None, 
                   nb_epoch=100, 
                   verbose=1):
        """
        Fit a model on a DataLoader
        """
        ## create regularizers
        if len(self._regularizers) > 0:
            regularizers = RegularizerModule(self._regularizers)
            regularizers.set_model(self)
        else:
            regularizers = None

        ## create constraints
        constraints = ConstraintModule(self._constraints)
        constraints.set_model(self)

        ## create callbacks
        if verbose > 0:
            self._callbacks += [TQDM()]
        callbacks = CallbackModule(self._callbacks)
        callbacks.set_model(self)

        callbacks.on_train_begin()

        for epoch_idx in range(nb_epoch):
            epoch_logs = {
                'nb_batches': int(math.ceil(len(loader.dataset.inputs)/loader.batch_size)),
                'nb_epoch': nb_epoch
            }
            callbacks.on_epoch_begin(epoch_idx, epoch_logs)

            for batch_idx,(x_batch, y_batch) in enumerate(loader):
                batch_logs = {
                    'batch_idx': batch_idx,
                    'batch_samples': len(x_batch)
                }                
                callbacks.on_batch_begin(batch_idx, batch_logs)

                inputs = Variable(x_batch)
                targets = Variable(y_batch)

                self._optimizer.zero_grad()
                outputs = self(inputs)
                loss = self._loss(outputs, targets)
                
                if regularizers is not None:
                    reg_loss = regularizers.compute_loss()
                    loss += reg_loss
                    batch_logs['reg_loss'] = reg_loss
                batch_logs['loss'] = loss.data[0]

                # make backward pass
                loss.backward()
                # make optimizer step to update weights
                self._optimizer.step()

                callbacks.on_batch_end(batch_idx, batch_logs)
                constraints.on_batch_end(batch_idx)

            if val_loader is not None:
                val_loss = self.evaluate_loader(val_loader)
                epoch_logs['val_loss'] = val_loss
            epoch_logs['loss'] = self.history.loss / self.history.samples_seen
            if regularizers is not None:
                epoch_logs['reg_loss'] = self.history.reg_loss / self.history.samples_seen

            callbacks.on_epoch_end(epoch_idx, epoch_logs)
            constraints.on_epoch_end(epoch_idx)
            if self.stop_training:
                break

        callbacks.on_train_end()

    def predict(self, 
                x, 
                batch_size=32, 
                verbose=1):
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size)
        preds = self.predict_loader(loader, verbose=verbose)
        return preds

    def predict_loader(self,
                       loader,
                       verbose=1):
        self.eval()
        preds = []
        for batch_idx, batch in enumerate(loader):
            if loader.dataset.has_target:
                batch = batch[0]
            x_batch = Variable(batch)
            batch_pred = self(x_batch)
            preds.append(batch_pred.data)
        self.train()
        return Variable(torch.cat(preds))

    def predict_on_batch(self, x):
        self.eval()
        x = Variable(x)
        preds = self(x)
        self.train()
        return preds

    def evaluate(self, 
                 x, 
                 y, 
                 batch_size=32, 
                 verbose=1):
        dataset = TensorDataset(x,y)
        loader = DataLoader(dataset, batch_size=batch_size)
        loss = self.evaluate_loader(loader)
        return loss

    def evaluate_loader(self, loader):
        self.eval()
        total_loss = 0.
        total_samples = 0.
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            x_batch = Variable(x_batch)
            y_batch = Variable(y_batch)

            y_pred = self(x_batch)
            loss = self._loss(y_pred, y_batch)
            total_loss += loss.data[0]*len(x_batch)
            total_samples += len(x_batch)
        self.train()
        return total_loss / total_samples

    def evaluate_on_batch(self, x, y):
        self.eval()
        x = Variable(x)
        y = Variable(y)
        y_pred = self(y)
        loss = self._loss(y_pred, y)
        self.train()
        return loss.data[0]

    def save_state_dict(self, file):
        """
        Save a model parameters to disk
        """
        # model parameters -> ordered dict
        state_dict = self.state_dict()
        torch.save(state_dict, file)












