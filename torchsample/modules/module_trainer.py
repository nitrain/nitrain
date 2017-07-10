"""
SuperModule for high level training on Pytorch models
"""
from __future__ import print_function
from __future__ import absolute_import

import math
from collections import OrderedDict

import torch as th
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# local imports
from ._utils import (_validate_loss_input, _validate_metric_input,
                     _validate_optimizer_input, _validate_initializer_input,
                     _standardize_user_data)

from ..callbacks import CallbackContainer, History, TQDM
#from ..regularizers import RegularizerContainer
#from ..initializers import InitializerContainer
#from ..constraints import ConstraintContainer
#from ..metrics import MetricsContainer


class ModuleTrainer(object):

    def __init__(self, model):
        """
        ModelTrainer for high-level training of Pytorch models

        Major Parts
        -----------
        - optimizer(s)
        - loss(es)
        - regularizers
        - initializers
        - constraints
        - metrics
        - callbacks
        """
        if not isinstance(model, nn.Module):
            raise ValueError('model argument must inherit from torch.nn.Module')
        self.model = model

        # callbacks
        self.history = History(self)
        self._callbacks = [self.history]

        # regularizers
        self._regularizers = []
        self._has_regularizers = False

        # initializers
        self._initializers = []

        # constraints
        self._constraints = []
        self._has_constraints = False

        # metrics
        self._metrics = []
        self._has_metrics = False

        # transforms
        self._transforms = []
        self._has_transforms = False

        # losses
        self._loss_fns = []
        self._has_multiple_loss_fns = False

        # other properties
        self._stop_training = False

    def forward(self, *input):
        """
        Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError('Subclass must implement this method')

    def set_loss(self, loss):
        self._loss = loss
        if not isinstance(loss, (list,tuple)):
            loss = [loss]
        loss = [_validate_loss_input(l) for l in loss]
        if len(loss) > 0:
            self._has_multiple_loss_fns = True
        self._loss_fns = loss

    def set_optimizer(self, optimizer, **kwargs):
        if type(optimizer) is type or isinstance(optimizer, str):
            if 'parameters' in kwargs:
                parameters = kwargs['parameters']
            else:
                parameters = self.model.parameters()

            optimizer = _validate_optimizer_input(optimizer)
            self._optimizer = optimizer(parameters, **kwargs)
        else:
            self._optimizer = optimizer

    def compile(self,
                optimizer,
                loss,
                regularizers=None,
                initializers=None,
                constraints=None,
                metrics=None,
                callbacks=None,
                transforms=None,
                **kwargs):
        opt_kwargs = {k.split('optimizer_')[1]:v for k,v in kwargs.items() if 'optimizer_' in k}
        self.set_optimizer(optimizer, **opt_kwargs)
        self.set_loss(loss)

    def fit(self,
            inputs,
            targets,
            nb_epoch=100,
            batch_size=32,
            shuffle=False,
            cuda_device=-1,
            verbose=1):
        """
        Fit a Pytorch model on given input and target tensor(s)

        Arguments
        ---------
        inputs : torch.Tensor or list/tuple of torch.Tensors
            input tensor(s)

        targets: torch.Tensor or list/tuple of torch.Tensors
            target tensor(s)

        val_data : 2-tuple/list or validation inputs + targets
            input and target tensor(s) to use for validation

        nb_epoch : integer
            number of epochs to train for

        batch_size : integer
            size of training batches

        shuffle : boolean
            if true, data will be randomly shuffled each epoch

        cuda_device : integer
            cuda device to put data on

        verbose : integer
            level of verbosity
        """
        inputs, targets = _standardize_user_data(inputs, targets)
        nb_targets = len(targets)
        nb_losses = len(self._loss_fns)
        if (nb_targets > nb_losses) and (nb_losses > 1):
            raise Exception('Must give either a) only one loss function or '
                            'b) one for every target')
        if nb_targets > nb_losses:
            self._loss_fns = [self._loss_fns[0]]*nb_targets

        if cuda_device > -1:
            inputs = [x.cuda(cuda_device) for x in inputs]
            targets = [y.cuda(cuda_device) for y in targets]
            self.model.cuda(cuda_device)

        # calculate total number of batches
        nb_batches = int(math.ceil(len(inputs[0]) / batch_size))

        # loop through each epoch
        for epoch_idx in range(nb_epoch):
            print('Epoch: ' , epoch_idx)
            tmp_losses = []
            # shuffle inputs and targets if necessary
            if shuffle:
                rand_indices = th.randperm(len(inputs[0]))
                inputs = [ins[rand_indices] for ins in inputs]
                targets = [tars[rand_indices] for tars in targets]

            # loop through each batch
            for batch_idx in range(nb_batches):

                # grab an input batch and a target batch if necessary
                input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size]) for x in inputs]
                target_batch = [Variable(y[batch_idx*batch_size:(batch_idx+1)*batch_size]) for y in targets]

                # zero grads and forward pass
                self._optimizer.zero_grad()
                output_batch = self.model(*input_batch)

                # apply multiple loss functions if necessary
                if not isinstance(output_batch, (list,tuple)):
                    output_batch = [output_batch]

                # multiple outputs, but they all go into one loss function
                loss = sum([self._loss_fns[loss_idx](output_batch[loss_idx], target_batch[loss_idx])
                                for loss_idx in range(nb_targets)])

                # backward pass and optimizer step
                loss.backward()
                self._optimizer.step()

                tmp_losses.append(loss.data[0])
            print('Epoch : ' , epoch_idx, ' : ' , np.mean(tmp_losses))

