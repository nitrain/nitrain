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
from ._utils import (validate_loss_input, validate_metric_input, 
                     validate_optimizer_input)
from ..callbacks import CallbackModule, History, TQDM
from ..constraints import ConstraintModule
from ..metrics import MetricsModule
from ..regularizers import RegularizerModule

class SuperModule(nn.Module):

    def __init__(self):
        """
        SuperModule for high-level training of Pytorch models

        TODO:
            - actually do something when shuffle=True on fit()
        OTHER:
            - add more metrics
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
        if not isinstance(loss, (list,tuple)):
            loss = [loss]
        loss = [validate_loss_input(l) for l in loss]
        if len(loss) > 0:
            self._has_multiple_loss_fns = True
        self._loss_fns = loss

    def set_optimizer(self, optimizer, **kwargs):
        if 'parameters' in kwargs:
            parameters = kwargs['parameters']
        else:
            parameters = self.parameters()

        optimizer = validate_optimizer_input(optimizer)
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
        if not isinstance(metrics, (list,tuple)):
            metrics = [metrics]
        metrics = [validate_metric_input(m) for m in metrics]
        self._has_metrics = True
        self._metrics = metrics

    def add_metric(self, metric):
        self._has_metrics = True
        self._metrics.append(validate_metric_input(metric))

    def fit(self,
            inputs, 
            targets=None,
            validation_data=None,
            nb_epoch=100, 
            batch_size=32,
            shuffle=False,
            cuda_device=-1,
            metrics=None,
            verbose=1):
        if not isinstance(inputs, (list,tuple)):
            inputs = [inputs]

        if targets is None:
            has_target = False
        else:
            has_target = True
            if not isinstance(targets, (list,tuple)):
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
        if metrics is not None:
            self.set_metrics(metrics)
        if self._has_metrics:
            metrics = MetricsModule(self._metrics)

        ## create callbacks
        with TQDM() as pbar:
            progressbar = []
            if verbose > 0:
                progressbar = [pbar]
            callbacks = CallbackModule(self._callbacks + progressbar)
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
                    batch_logs = {'batch_idx': batch_idx}  
                    callbacks.on_batch_begin(batch_idx, batch_logs) 

                    input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size]) for x in inputs]
                    if has_target:
                        target_batch = [Variable(y[batch_idx*batch_size:(batch_idx+1)*batch_size]) for y in targets]

                    if cuda_device > -1:
                        input_batch = [ins.cuda(cuda_device) for ins in input_batch]
                        if has_target:
                            target_batch = [targs.cuda(cuda_device) for targs in target_batch]

                    batch_logs['batch_samples'] = len(input_batch[0])

                    ## ZERO GRAD AND FORWARD PASS
                    self._optimizer.zero_grad()
                    outputs = self(*input_batch)

                    if not isinstance(outputs, (list,tuple)):
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
                    if self._has_metrics:
                        val_loss, val_metric_logs = val_loss
                        epoch_logs.update(val_metric_logs)
                    epoch_logs['val_loss'] = val_loss
                    self.history.batch_metrics['val_loss'] = val_loss
                
                # END OF EPOCH
                epoch_logs.update(self.history.batch_metrics)
                if self._has_metrics:
                    epoch_logs.update(metric_logs)

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
                   metrics=None,
                   verbose=1):
        if val_loader is None:
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
        if metrics is not None:
            self.set_metrics(metrics)
        if self._has_metrics:
            metrics = MetricsModule(self._metrics)

        ## create callbacks
        if verbose > 0:
            self._callbacks += [TQDM()]
        callbacks = CallbackModule(self._callbacks)
        callbacks.set_model(self)

        callbacks.on_train_begin()

        nb_batches = int(math.ceil(len(loader.dataset) / loader.batch_size))
        for epoch_idx in range(nb_epoch):
            epoch_logs = {
                'nb_batches': nb_batches,
                'nb_epoch': nb_epoch,
                'has_validation_data': has_validation_data
            }
            callbacks.on_epoch_begin(epoch_idx, epoch_logs)

            for batch_idx, batch_data in enumerate(loader):
                batch_logs = {'batch_idx': batch_idx}  
                callbacks.on_batch_begin(batch_idx, batch_logs) 

                if len(batch_data) == 1:
                    # no target
                    input_batch = batch_data[0]
                    has_target = False
                elif len(batch_data) == 2:
                    input_batch = batch_data[0]
                    target_batch = batch_data[1]
                    has_target = True
                if not isinstance(input_batch, (list,tuple)):
                    input_batch = [input_batch]
                input_batch = [Variable(ins) for ins in input_batch]
                if has_target:
                    if not isinstance(target_batch, (list,tuple)):
                        target_batch = [target_batch]
                    target_batch = [Variable(targs) for targs in target_batch]
                    nb_targets = len(target_batch)

                if cuda_device > -1:
                    input_batch = [ins.cuda(cuda_device) for ins in input_batch]
                    if has_target:
                        target_batch = [targs.cuda(cuda_device) for targs in target_batch]

                batch_logs['batch_samples'] = len(input_batch[0])

                ## ZERO GRAD AND FORWARD PASS
                self._optimizer.zero_grad()
                outputs = self(*input_batch)

                if not isinstance(outputs, (list,tuple)):
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
                val_loss = self.evaluate_loader(val_loader,
                                                cuda_device=cuda_device)
                if self._has_metrics:
                    val_loss, val_metric_logs = val_loss
                    epoch_logs.update(val_metric_logs)
                epoch_logs['val_loss'] = val_loss
                self.history.batch_metrics['val_loss'] = val_loss
            
            # END OF EPOCH
            epoch_logs.update(self.history.batch_metrics)
            if self._has_metrics:
                epoch_logs.update(metric_logs)

            callbacks.on_epoch_end(epoch_idx, epoch_logs)

            if self._has_constraints:
                constraints.on_epoch_end(epoch_idx)
            if self._has_metrics:
                metrics.reset()
            if self._stop_training:
                break

        callbacks.on_train_end()

    def train_on_batch(self, 
                       inputs, 
                       targets=None,
                       regularizers=None,
                       constraints=None,
                       callbacks=None,
                       cuda_device=-1):
        if not isinstance(inputs, (list,tuple)):
            inputs = [inputs]
        if targets is not None:
            if not isinstance(targets, (list,tuple)):
                targets = [targets]
            has_target = True
            nb_targets = len(targets)
        else:
            has_target = False

        input_batch = [Variable(x) for x in inputs]
        if has_target:
            target_batch = [Variable(y) for y in targets]

        if cuda_device > -1:
            input_batch = [ins.cuda(cuda_device) for ins in input_batch]
            if has_target:
                target_batch = [targs.cuda(cuda_device) for targs in target_batch]
         
        ## ZERO GRAD AND FORWARD PASS
        self._optimizer.zero_grad()
        outputs = self(*input_batch)

        if not isinstance(outputs, (list,tuple)):
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

        if self._has_constraints and constraints.has_lagrangian:
            constraint_loss = constraints(self)
            loss += constraint_loss

        # BACKWARD PASS AND OPTIMIZER STEP
        loss.backward()
        self._optimizer.step()

    def predict(self, 
                inputs, 
                batch_size=32,
                cuda_device=-1, 
                verbose=1):
        if not isinstance(inputs, (list,tuple)):
            inputs = [inputs]

        nb_batches = int(math.ceil(inputs[0].size(0) / batch_size))
        prediction_list = []
        for batch_idx in range(nb_batches):
            input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size]) for x in inputs]

            if cuda_device > -1:
                input_batch = [ins.cuda(cuda_device) for ins in input_batch]

            prediction_list.append(self(*input_batch))
        return torch.cat(prediction_list,0)

    def predict_loader(self,
                       loader,
                       cuda_device=-1,
                       verbose=1):
        prediction_list = []
        for batch_idx, batch_data in enumerate(loader):
            if not isinstance(batch_data, (tuple,list)):
                batch_data = [batch_data]
            input_batch = batch_data[0]
            if not isinstance(input_batch, (list,tuple)):
                input_batch = [input_batch]
            input_batch = [Variable(ins) for ins in input_batch]
            if cuda_device > -1:
                input_batch = [ins.cuda(cuda_device) for ins in input_batch]

            prediction_list.append(self(*input_batch))
        return torch.cat(prediction_list,0)

    def predict_on_batch(self, 
                         inputs, 
                         cuda_device=-1):
        if not isinstance(inputs, (list,tuple)):
            inputs = [inputs]
        if cuda_device > -1:
            inputs = [ins.cuda(cuda_device) for ins in inputs]
        preds = self(*inputs)
        return preds

    def evaluate(self, 
                 inputs, 
                 targets=None, 
                 batch_size=32,
                 cuda_device=-1, 
                 verbose=1):
        # put model in evaluation mode
        self.eval()
        if not isinstance(inputs, (list,tuple)):
            inputs = [inputs]

        if targets is None:
            has_target = False
        else:
            has_target = True
            if not isinstance(targets, (list,tuple)):
                targets = [targets]
            nb_targets = len(targets)

        if self._has_metrics:
            metrics = MetricsModule(self._metrics, prefix='val_')

        total_loss = 0.
        total_samples = 0.
        nb_batches = int(math.ceil(inputs[0].size(0) / batch_size))
        for batch_idx in range(nb_batches):
            input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size]) for x in inputs]
            if has_target:
                target_batch = [Variable(y[batch_idx*batch_size:(batch_idx+1)*batch_size]) for y in targets]

            if cuda_device > -1:
                input_batch = [ins.cuda(cuda_device) for ins in input_batch]
                if has_target:
                    target_batch = [targs.cuda(cuda_device) for targs in target_batch]

            outputs = self(*input_batch)
            if not isinstance(outputs, (list,tuple)):
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
            if self._has_metrics:
                metric_logs = metrics(outputs[0], target_batch[0])

            total_loss += loss.data[0]*len(input_batch[0])
            total_samples += len(input_batch[0])
        # put model back in training mode
        self.train()
        if self._has_metrics:
            return total_loss / float(total_samples), metric_logs
        else:
            return total_loss / float(total_samples)

    def evaluate_loader(self, 
                        loader, 
                        cuda_device=-1):
        self.eval()
        if self._has_metrics:
            metrics = MetricsModule(self._metrics, prefix='val_')

        total_loss = 0.
        total_samples = 0.
        for batch_idx, batch_data in enumerate(loader):
            if len(batch_data) == 1:
                # no target
                input_batch = batch_data[0]
                has_target = False
            elif len(batch_data) == 2:
                input_batch = batch_data[0]
                target_batch = batch_data[1]
                has_target = True
            if not isinstance(input_batch, (list,tuple)):
                input_batch = [input_batch]
            input_batch = [Variable(ins) for ins in input_batch]
            if has_target:
                if not isinstance(target_batch, (list,tuple)):
                    target_batch = [target_batch]
                target_batch = [Variable(targs) for targs in target_batch]
                nb_targets = len(target_batch)

            if cuda_device > -1:
                input_batch = [ins.cuda(cuda_device) for ins in input_batch]
                if has_target:
                    target_batch = [targs.cuda(cuda_device) for targs in target_batch]

            outputs = self(*input_batch)

            if not isinstance(outputs, (list,tuple)):
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

            if self._has_metrics:
                metric_logs = metrics(outputs[0], target_batch[0])

            total_loss += loss.data[0]*len(input_batch[0])
            total_samples += len(input_batch[0])

        self.train()
        if self._has_metrics:
            return total_loss / float(total_samples), metric_logs
        else:
            return total_loss / float(total_samples)

    def evaluate_on_batch(self, 
                          inputs, 
                          targets, 
                          cuda_device=-1):
        self.eval()
        if not isinstance(inputs, (list,tuple)):
            inputs = [inputs]

        if targets is None:
            has_target = False
        else:
            has_target = True
            if not isinstance(targets, (list,tuple)):
                targets = [targets]
            nb_targets = len(targets)

        if len(self._loss_fns) > 1:
            has_multiple_loss_fns = True
        else:
            has_multiple_loss_fns = False

        input_batch = [Variable(x) for x in inputs]
        if has_target:
            target_batch = [Variable(y) for y in targets]

        if cuda_device > -1:
            input_batch = [ins.cuda(cuda_device) for ins in input_batch]
            if has_target:
                target_batch = [targs.cuda(cuda_device) for targs in target_batch]

        outputs = self(*input_batch)
        if not isinstance(outputs, (list,tuple)):
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







