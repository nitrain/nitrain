"""
SuperModule for high level training on Pytorch models

NOTES
-----
- only supporting one loss function right now
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings
import math
from collections import OrderedDict

import torch as th
import torch.nn as nn
from torch.autograd import Variable

# local imports
from ._utils import (_validate_loss_input, _validate_metric_input,
                     _validate_optimizer_input, _validate_initializer_input,
                     _standardize_user_data, _parse_num_inputs_and_targets)

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
        if isinstance(loss, (tuple, list)):
            self._loss_fn = [_validate_loss_input(l) for l in loss]
        else:
            self._loss_fn = _validate_loss_input(loss)

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
            targets=None,
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
        num_inputs, num_targets = _parse_num_inputs_and_targets(inputs, targets)

        if (num_inputs == 1) and (num_targets == 1):
            self._fit_single_input_single_target(inputs,
                                                 targets,
                                                 nb_epoch,
                                                 batch_size,
                                                 shuffle,
                                                 cuda_device,
                                                 verbose)

        elif (num_inputs == 1) and (num_targets > 1):
            # use same loss function for all targets if multiple loss fns not explicitly given
            if not isinstance(self._loss_fn, (tuple, list)):
                self._loss_fn = [self._loss_fn] * num_targets
            else:
                if len(self._loss_fn) != num_targets:
                    raise ValueError('must give one loss function for every input if you give multiple')

            self._fit_single_input_multi_target(inputs,
                                                targets,
                                                nb_epoch,
                                                batch_size,
                                                shuffle,
                                                cuda_device,
                                                verbose)

        elif (num_inputs == 1) and (num_targets == 0):
            self._fit_single_input_no_target(inputs,
                                             nb_epoch,
                                             batch_size,
                                             shuffle,
                                             cuda_device,
                                             verbose)

        elif (num_inputs > 1) and (num_targets == 1):
            self._fit_multi_input_single_target(inputs,
                                                targets,
                                                nb_epoch,
                                                batch_size,
                                                shuffle,
                                                cuda_device,
                                                verbose)

        elif (num_inputs > 1) and (num_targets > 1):
            # use same loss function for all targets if multiple loss fns not explicitly given
            if not isinstance(self._loss_fn, (tuple, list)):
                self._loss_fn = [self._loss_fn] * num_targets
            else:
                if len(self._loss_fn) != num_targets:
                    raise ValueError('must give one loss function for every input if you give multiple')

            self._fit_multi_input_multi_target(inputs,
                                               targets,
                                               nb_epoch,
                                               batch_size,
                                               shuffle,
                                               cuda_device,
                                               verbose)

        elif (num_inputs > 1) and (num_targets == 0):
            self._fit_multi_input_no_target(inputs,
                                            nb_epoch,
                                            batch_size,
                                            shuffle,
                                            cuda_device,
                                            verbose)


    def _fit_single_input_single_target(self,
                                        inputs,
                                        targets,
                                        nb_epoch,
                                        batch_size,
                                        shuffle,
                                        cuda_device,
                                        verbose):
        print('Fitting single input, single target')
        if cuda_device > -1:
            inputs = inputs.cuda(cuda_device)
            targets = targets.cuda(cuda_device)
            self.model.cuda(cuda_device)

        nb_batches = int(math.ceil(len(inputs) / batch_size))

        for epoch_idx in range(nb_epoch):
            print('Epoch: ' , epoch_idx)
            tmp_losses = []
            # shuffle inputs and targets if necessary
            if shuffle:
                rand_indices = th.randperm(len(inputs))
                inputs = inputs[rand_indices]
                targets = targets[rand_indices]

            # loop through each batch
            for batch_idx in range(nb_batches):

                # grab an input batch and a target batch if necessary
                input_batch = Variable(inputs[batch_idx*batch_size:(batch_idx+1)*batch_size])
                target_batch = Variable(targets[batch_idx*batch_size:(batch_idx+1)*batch_size])

                # zero grads and forward pass
                self._optimizer.zero_grad()
                output_batch = self.model(input_batch)

                # multiple outputs, but they all go into one loss function
                loss = self._loss_fn(output_batch, target_batch)

                # backward pass and optimizer step
                loss.backward()
                self._optimizer.step()

                tmp_losses.append(loss.data[0])
            print('Epoch : ' , epoch_idx, ' : ' , th.mean(th.FloatTensor(tmp_losses)))


    def _fit_single_input_multi_target(self,
                                       inputs,
                                       targets,
                                       nb_epoch,
                                       batch_size,
                                       shuffle,
                                       cuda_device,
                                       verbose):
        print('Fitting single input, multi target')
        if cuda_device > -1:
            inputs = inputs.cuda(cuda_device)
            targets = [target_.cuda(cuda_device) for target_ in targets]
            self.model.cuda(cuda_device)

        nb_batches = int(math.ceil(len(inputs) / batch_size))

        for epoch_idx in range(nb_epoch):
            print('Epoch: ' , epoch_idx)
            tmp_losses = []
            # shuffle inputs and targets if necessary
            if shuffle:
                rand_indices = th.randperm(len(inputs))
                inputs = inputs[rand_indices]
                targets = [target_[rand_indices] for target_ in targets]

            # loop through each batch
            for batch_idx in range(nb_batches):

                # grab an input batch and a target batch if necessary
                input_batch = Variable(inputs[batch_idx*batch_size:(batch_idx+1)*batch_size])
                target_batch = [Variable(target_[batch_idx*batch_size:(batch_idx+1)*batch_size])
                                for target_ in targets]

                # zero grads and forward pass
                self._optimizer.zero_grad()
                output_batch = self.model(input_batch)

                loss = sum([self._loss_fn[idx](output_batch[idx], target_batch[idx]) for idx in range(len(output_batch))])

                # backward pass and optimizer step
                loss.backward()
                self._optimizer.step()

                tmp_losses.append(loss.data[0])
            print('Epoch : ' , epoch_idx, ' : ' , th.mean(th.FloatTensor(tmp_losses)))

    def _fit_multi_input_single_target(self,
                                       inputs,
                                       targets,
                                       nb_epoch,
                                       batch_size,
                                       shuffle,
                                       cuda_device,
                                       verbose):
        print('Fitting multi input, single target')
        if cuda_device > -1:
            inputs = [input_.cuda(cuda_device) for input_ in inputs] 
            targets = targets.cuda(cuda_device)
            self.model.cuda(cuda_device)

        nb_batches = int(math.ceil(len(inputs[0]) / batch_size))

        for epoch_idx in range(nb_epoch):
            print('Epoch: ' , epoch_idx)
            tmp_losses = []
            # shuffle inputs and targets if necessary
            if shuffle:
                rand_indices = th.randperm(len(inputs[0]))
                inputs = [input_[rand_indices] for input_ in inputs]
                targets = targets[rand_indices]

            # loop through each batch
            for batch_idx in range(nb_batches):

                # grab an input batch and a target batch if necessary
                input_batch = [Variable(input_[batch_idx*batch_size:(batch_idx+1)*batch_size])
                               for input_ in inputs]
                target_batch = Variable(targets[batch_idx*batch_size:(batch_idx+1)*batch_size])

                # zero grads and forward pass
                self._optimizer.zero_grad()
                output_batch = self.model(*input_batch)

                # multiple outputs, but they all go into one loss function
                loss = self._loss_fn(output_batch, target_batch)

                # backward pass and optimizer step
                loss.backward()
                self._optimizer.step()

                tmp_losses.append(loss.data[0])
            print('Epoch : ' , epoch_idx, ' : ' , th.mean(th.FloatTensor(tmp_losses)))

    def _fit_multi_input_multi_target(self,
                                       inputs,
                                       targets,
                                       nb_epoch,
                                       batch_size,
                                       shuffle,
                                       cuda_device,
                                       verbose):
        print('Fitting multi input, multi target')
        if cuda_device > -1:
            inputs = [input_.cuda(cuda_device) for input_ in inputs] 
            targets = targets.cuda(cuda_device)
            self.model.cuda(cuda_device)

        nb_batches = int(math.ceil(len(inputs[0]) / batch_size))

        for epoch_idx in range(nb_epoch):
            print('Epoch: ' , epoch_idx)
            tmp_losses = []
            # shuffle inputs and targets if necessary
            if shuffle:
                rand_indices = th.randperm(len(inputs[0]))
                inputs = [input_[rand_indices] for input_ in inputs]
                targets = targets[rand_indices]

            # loop through each batch
            for batch_idx in range(nb_batches):

                # grab an input batch and a target batch if necessary
                input_batch = [Variable(input_[batch_idx*batch_size:(batch_idx+1)*batch_size])
                               for input_ in inputs]
                target_batch = [Variable(target_[batch_idx*batch_size:(batch_idx+1)*batch_size])
                               for target_ in targets]
                # zero grads and forward pass
                self._optimizer.zero_grad()
                output_batch = self.model(*input_batch)

                # multiple outputs, but they all go into one loss function
                loss = sum([self._loss_fn[idx](output_batch[idx], target_batch[idx]) for idx in range(len(output_batch))])

                # backward pass and optimizer step
                loss.backward()
                self._optimizer.step()

                tmp_losses.append(loss.data[0])
            print('Epoch : ' , epoch_idx, ' : ' , th.mean(th.FloatTensor(tmp_losses)))

    def _fit_single_input_no_target(self,
                                    inputs,
                                    nb_epoch,
                                    batch_size,
                                    shuffle,
                                    cuda_device,
                                    verbose):
        print('Fitting single input, no target')
        if cuda_device > -1:
            inputs = inputs.cuda(cuda_device)
            self.model.cuda(cuda_device)

        nb_batches = int(math.ceil(len(inputs) / batch_size))

        for epoch_idx in range(nb_epoch):
            print('Epoch: ' , epoch_idx)
            tmp_losses = []
            # shuffle inputs and targets if necessary
            if shuffle:
                rand_indices = th.randperm(len(inputs))
                inputs = inputs[rand_indices]

            # loop through each batch
            for batch_idx in range(nb_batches):

                # grab an input batch and a target batch if necessary
                input_batch = Variable(inputs[batch_idx*batch_size:(batch_idx+1)*batch_size])

                # zero grads and forward pass
                self._optimizer.zero_grad()
                output_batch = self.model(input_batch)

                # multiple outputs, but they all go into one loss function
                loss = self._loss_fn(output_batch)

                # backward pass and optimizer step
                loss.backward()
                self._optimizer.step()

                tmp_losses.append(loss.data[0])
            print('Epoch : ' , epoch_idx, ' : ' , th.mean(th.FloatTensor(tmp_losses)))

    def _fit_multi_input_no_target(self,
                                   inputs,
                                   nb_epoch,
                                   batch_size,
                                   shuffle,
                                   cuda_device,
                                   verbose):
        print('Fitting multi input, no target')
        if cuda_device > -1:
            inputs = [input_.cuda(cuda_device) for input_ in inputs]
            self.model.cuda(cuda_device)

        nb_batches = int(math.ceil(len(inputs[0]) / batch_size))

        for epoch_idx in range(nb_epoch):
            print('Epoch: ' , epoch_idx)
            tmp_losses = []
            # shuffle inputs and targets if necessary
            if shuffle:
                rand_indices = th.randperm(len(inputs[0]))
                inputs = [input_[rand_indices] for input_ in inputs]

            # loop through each batch
            for batch_idx in range(nb_batches):

                # grab an input batch and a target batch if necessary
                input_batch = [Variable(input_[batch_idx*batch_size:(batch_idx+1)*batch_size])
                               for input_ in inputs]

                # zero grads and forward pass
                self._optimizer.zero_grad()
                output_batch = self.model(*input_batch)

                # multiple outputs, but they all go into one loss function
                loss = self._loss_fn(output_batch)

                # backward pass and optimizer step
                loss.backward()
                self._optimizer.step()

                tmp_losses.append(loss.data[0])
            print('Epoch : ' , epoch_idx, ' : ' , th.mean(th.FloatTensor(tmp_losses)))
    def predict(self,
                inputs,
                batch_size=32,
                cuda_device=-1,
                verbose=1):
        inputs = _standardize_user_data(inputs)

        if cuda_device >= 0:
            inputs = [x.cuda(cuda_device) for x in inputs]
            self.model.cuda(cuda_device)

        nb_batches = int(math.ceil(inputs[0].size(0) / batch_size))
        prediction_list = []
        for batch_idx in range(nb_batches):
            input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size]) for x in inputs]
            prediction_list.append(self.model(*input_batch))
        return th.cat(prediction_list,0)

