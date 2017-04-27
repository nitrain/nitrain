"""
SuperModule for high level training on Pytorch models
"""
from __future__ import print_function
from __future__ import absolute_import

import math

import torch
import torch.nn as nn
from torch.autograd import Variable

# local imports
from ..callbacks import CallbackModule, History, TQDM
from ..constraints import ConstraintModule
from ..metrics import MetricsModule
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

        # callbacks
        self.history = History()
        self._callbacks = [self.history]
        # constraints
        self._constraints = []
        self._has_constraints = False
        # regularizers
        self._regularizers = []
        self._has_regularizers = False
        # metrics
        self._metrics = []
        self._has_metrics = False
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
        if not isinstance(loss, list) and not isinstance(loss, tuple):
            loss = [loss]
        if len(loss) > 0:
            self._has_multiple_loss_fns = True
        self._loss_fns = loss

    def set_optimizer(self, optimizer, **kwargs):
        if 'parameters' in kwargs:
            parameters = kwargs['parameters']
        else:
            parameters = self.parameters()
        self._optimizer = optimizer(parameters, **kwargs)

    def set_regularizers(self, regularizers):
        self._has_regularizers = True
        self._regularizers = regularizers

    def add_regularizer(self, regularizer):
        self._has_regularizers = True
        self._regularizers.append(regularizer)

    def set_constraints(self, constraints):
        self._has_constraints = True
        self._constraints = constraints

    def add_constraint(self, constraint):
        self._has_constraints = True
        self._constraints.append(constraint)

    def set_callbacks(self, callbacks):
        self._callbacks += callbacks

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def set_metrics(self, metrics):
        self._has_metrics = True
        self._metrics = metrics

    def add_metric(self, metric):
        self._has_metrics = True
        self._metrics.append(metric)

    def fit(self,
            inputs, 
            targets=None,
            validation_data=None,
            nb_epoch=100, 
            batch_size=32,
            shuffle=False,
            cuda_device=-1,
            verbose=1):
        if not isinstance(inputs, list):
            inputs = [inputs]

        if targets is None:
            has_target = False
        else:
            has_target = True
            if not isinstance(targets, list):
                targets = [targets]
            nb_targets = len(targets)

        if validation_data is None:
            has_validation_data = False
        else:
            has_validation_data = True

        ## create regularizers
        if self._has_regularizers:
            regularizers = RegularizerModule(self._regularizers)

        ## create constraints
        if self._has_constraints:
            constraints = ConstraintModule(self._constraints)
            constraints.set_model(self)

        ## create metrics
        if self._has_metrics:
            metrics = MetricsModule(self._metrics)

        ## create callbacks
        if verbose > 0:
            self._callbacks += [TQDM()]
        callbacks = CallbackModule(self._callbacks)
        callbacks.set_model(self)

        callbacks.on_train_begin()

        nb_batches = int(math.ceil(inputs[0].size(0) / batch_size))
        for epoch_idx in range(nb_epoch):
            epoch_logs = {
                'nb_batches': nb_batches,
                'nb_epoch': nb_epoch,
                'has_validation_data': has_validation_data
            }
            callbacks.on_epoch_begin(epoch_idx, epoch_logs)

            for batch_idx in range(nb_batches):
                input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size]) for x in inputs]
                if has_target:
                    target_batch = [Variable(y[batch_idx*batch_size:(batch_idx+1)*batch_size]) for y in targets]

                if cuda_device > 0:
                    input_batch = [ins.cuda(cuda_device) for ins in input_batch]
                    if has_target:
                        target_batch = [targs.cuda(cuda_device) for targs in target_batch]

                batch_logs = {
                    'batch_idx': batch_idx,
                    'batch_samples': len(input_batch[0])
                }                
                callbacks.on_batch_begin(batch_idx, batch_logs)

                ## ZERO GRAD AND FORWARD PASS
                self._optimizer.zero_grad()
                outputs = self(*input_batch)

                if not isinstance(outputs, list) and not isinstance(outputs, tuple):
                    outputs = [outputs]
                if has_target:
                    loss = self._loss_fns[0](outputs[0], target_batch[0])
                    for loss_idx in range(1,nb_targets):
                        if self._has_multiple_loss_fns:
                            loss += self._loss_fns[loss_idx](outputs[loss_idx], target_batch[loss_idx])
                        else:
                            loss += self._loss_fns[0](outputs[loss_idx], target_batch[loss_idx])
                else:
                    loss = self._loss_fns[0](outputs[0])
                    for loss_idx in range(1,nb_targets):
                        if self._has_multiple_loss_fns:
                            loss += self._loss_fns[loss_idx](outputs[loss_idx])
                        else:
                            loss += self._loss_fns[0](outputs[loss_idx])
                
                if self._has_regularizers:
                    regularizer_loss = regularizers(self)
                    loss += regularizer_loss
                    batch_logs['regularizer_loss'] = regularizer_loss

                if self._has_constraints and constraints.has_lagrangian:
                    constraint_loss = constraints(self)
                    loss += constraint_loss
                    batch_logs['constraint_loss'] = constraint_loss

                batch_logs['loss'] = loss.data[0]

                if self._has_metrics:
                    metric_logs = metrics(outputs[0], target_batch[0])
                    batch_logs.update(metric_logs)

                # BACKWARD PASS AND OPTIMIZER STEP
                loss.backward()
                self._optimizer.step()

                callbacks.on_batch_end(batch_idx, batch_logs)
                if self._has_constraints:
                    constraints.on_batch_end(batch_idx)

            if has_validation_data:
                val_loss = self.evaluate(*validation_data, 
                                         batch_size=batch_size,
                                         cuda_device=cuda_device)
                epoch_logs['val_loss'] = val_loss

            # END OF EPOCH
            epoch_logs.update(self.history.batch_metrics)
            if self._has_metrics:
                epoch_logs.update(metrics.get_logs())

            callbacks.on_epoch_end(epoch_idx, epoch_logs)

            if self._has_constraints:
                constraints.on_epoch_end(epoch_idx)
            if self._has_metrics:
                metrics.reset()
            if self._stop_training:
                break

        callbacks.on_train_end()

    def fit_loader(self, 
                   loader, 
                   val_loader=None, 
                   nb_epoch=100,
                   cuda_device=-1,
                   verbose=1):
        """
        Fit a model on a DataLoader
        """
        ## create regularizers
        if len(self._regularizers) > 0:
            regularizers = RegularizerModule(self._regularizers)
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

            for batch_idx, (x_batch, y_batch) in enumerate(loader):
                batch_logs = {
                    'batch_idx': batch_idx,
                    'batch_samples': len(x_batch),
                }                
                callbacks.on_batch_begin(batch_idx, batch_logs)

                inputs = Variable(x_batch)
                targets = Variable(y_batch)
                if cuda_device is not None:
                    inputs = inputs.cuda(cuda_device)
                    targets = targets.cuda(cuda_device)

                self._optimizer.zero_grad()
                outputs = self(inputs)
                loss = self._loss(outputs, targets)
                
                if regularizers is not None:
                    reg_loss = regularizers(self)
                    loss += reg_loss
                    batch_logs['reg_loss'] = reg_loss

                if constraints is not None and constraints.has_lagrangian:
                    constraint_loss = constraints(self)
                    loss += constraint_loss
                    batch_logs['constraint_loss'] = constraint_loss

                batch_logs['loss'] = loss.data[0]

                # make backward pass
                loss.backward()
                # make optimizer step to update weights
                self._optimizer.step()

                callbacks.on_batch_end(batch_idx, batch_logs)
                constraints.on_batch_end(batch_idx)

            if val_loader is not None:
                val_loss = self.evaluate_loader(val_loader, 
                                                cuda_device=cuda_device)
                epoch_logs['val_loss'] = val_loss
            epoch_logs['loss'] = self.history.loss / self.history.samples_seen
            if regularizers is not None:
                epoch_logs['reg_loss'] = self.history.reg_loss / self.history.samples_seen
            if constraints is not None and constraints.has_lagrangian:
                epoch_logs['constraint_loss'] = self.history.constraint_loss / self.history.samples_seen
            callbacks.on_epoch_end(epoch_idx, epoch_logs)
            constraints.on_epoch_end(epoch_idx)
            if self._stop_training:
                break

        callbacks.on_train_end()

    def fit_on_batch(self, 
                     x, 
                     y, 
                     cuda_device=-1):
        inputs = Variable(x)
        targets = Variable(y)
        if cuda_device is not None:
            inputs = inputs.cuda(cuda_device)
            targets = targets.cuda(cuda_device)

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

    def predict(self, 
                inputs, 
                batch_size=32,
                cuda_device=-1, 
                verbose=1):
        if not isinstance(inputs, list):
            inputs = [inputs]

        nb_batches = int(math.ceil(inputs[0].size(0) / batch_size))
        output_list = []
        for batch_idx in range(nb_batches):
            input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size]) for x in inputs]

            if cuda_device > 0:
                input_batch = [ins.cuda(cuda_device) for ins in input_batch]

            output_list.append(self(*input_batch))
        return torch.cat(output_list,0)

    def predict_loader(self,
                       loader,
                       cuda_device=-1,
                       verbose=1):
        self.eval()
        preds = []
        for batch_idx, batch in enumerate(loader):
            if loader.dataset.has_target:
                batch = batch[0]
            x_batch = Variable(batch)
            if cuda_device is not None:
                x_batch = x_batch.cuda(cuda_device)
            batch_pred = self(x_batch)
            preds.append(batch_pred.data)
        self.train()
        return Variable(torch.cat(preds))

    def predict_on_batch(self, 
                         x, 
                         cuda_device=-1):
        self.eval()
        x = Variable(x)
        if cuda_device > 0:
            x = x.cuda(cuda_device)
        preds = self(x)
        self.train()
        return preds

    def evaluate(self, 
                 inputs, 
                 targets=None, 
                 batch_size=32,
                 cuda_device=-1, 
                 verbose=1):
        # put model in evaluation mode
        self.eval()
        if not isinstance(inputs, list):
            inputs = [inputs]

        if targets is None:
            has_target = False
        else:
            has_target = True
            if not isinstance(targets, list):
                targets = [targets]
            nb_targets = len(targets)

        total_loss = 0.
        total_samples = 0.
        nb_batches = int(math.ceil(inputs[0].size(0) / batch_size))
        for batch_idx in range(nb_batches):
            input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size]) for x in inputs]
            if has_target:
                target_batch = [Variable(y[batch_idx*batch_size:(batch_idx+1)*batch_size]) for y in targets]

            if cuda_device > 0:
                input_batch = [ins.cuda(cuda_device) for ins in input_batch]
                if has_target:
                    target_batch = [targs.cuda(cuda_device) for targs in target_batch]

            outputs = self(*input_batch)
            if not isinstance(outputs, list) and not isinstance(outputs, tuple):
                outputs = [outputs]
            if has_target:
                loss = self._loss_fns[0](outputs[0], target_batch[0])
                for loss_idx in range(1,nb_targets):
                    if self._has_multiple_loss_fns:
                        loss += self._loss_fns[loss_idx](outputs[loss_idx], target_batch[loss_idx])
                    else:
                        loss += self._loss_fns[0](outputs[loss_idx], target_batch[loss_idx])
            else:
                loss = self._loss_fns[0](outputs[0])
                for loss_idx in range(1,nb_targets):
                    if self._has_multiple_loss_fns:
                        loss += self._loss_fns[loss_idx](outputs[loss_idx])
                    else:
                        loss += self._loss_fns[0](outputs[loss_idx])

            total_loss += loss.data[0]*len(input_batch[0])
            total_samples += len(input_batch[0])
        # put model back in training mode
        self.train()
        return total_loss / total_samples

    def evaluate_loader(self, 
                        loader, 
                        cuda_device=-1):
        self.eval()
        total_loss = 0.
        total_samples = 0.
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            x_batch = Variable(x_batch)
            y_batch = Variable(y_batch)
            if cuda_device > 0:
                x_batch = x_batch.cuda(cuda_device)
                y_batch = y_batch.cuda(cuda_device)

            y_pred = self(x_batch)
            loss = self._loss(y_pred, y_batch)
            total_loss += loss.data[0]*len(x_batch)
            total_samples += len(x_batch)
        self.train()
        return total_loss / total_samples

    def evaluate_on_batch(self, 
                          inputs, 
                          targets, 
                          cuda_device=-1):
        self.eval()
        if not isinstance(inputs, list):
            inputs = [inputs]

        if targets is None:
            has_target = False
        else:
            has_target = True
            if not isinstance(targets, list):
                targets = [targets]
            nb_targets = len(targets)

        if len(self._loss_fns) > 1:
            has_multiple_loss_fns = True
        else:
            has_multiple_loss_fns = False

        input_batch = [Variable(x) for x in inputs]
        if has_target:
            target_batch = [Variable(y) for y in targets]

        if cuda_device > 0:
            input_batch = [ins.cuda(cuda_device) for ins in input_batch]
            if has_target:
                target_batch = [targs.cuda(cuda_device) for targs in target_batch]

        outputs = self(*input_batch)
        if not isinstance(outputs, list) and not isinstance(outputs, tuple):
            outputs = [outputs]
        if has_target:
            loss = self._loss_fns[0](outputs[0], target_batch[0])
            for loss_idx in range(1,nb_targets):
                if has_multiple_loss_fns:
                    loss += self._loss_fns[loss_idx](outputs[loss_idx], target_batch[loss_idx])
                else:
                    loss += self._loss_fns[0](outputs[loss_idx], target_batch[loss_idx])
        else:
            loss = self._loss_fns[0](outputs[0])
            for loss_idx in range(1,nb_targets):
                if has_multiple_loss_fns:
                    loss += self._loss_fns[loss_idx](outputs[loss_idx])
                else:
                    loss += self._loss_fns[0](outputs[loss_idx])
        self.train()
        return loss.data[0]

    def save_state_dict(self, file):
        """
        Save a model parameters to disk
        """
        # model parameters -> ordered dict
        state_dict = self.state_dict()
        torch.save(state_dict, file)












