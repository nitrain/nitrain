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
import torch.backends.cudnn as cudnn

# local imports
from ._utils import (_validate_loss_input, _validate_metric_input, 
                     _validate_optimizer_input, _validate_initializer_input,
                     _get_current_time, _nb_function_args)
from ..callbacks import CallbackModule, History, TQDM
from ..constraints import ConstraintModule
from ..initializers import InitializerModule
from ..metrics import MetricsModule
from ..regularizers import RegularizerModule


class ModuleTrainer(object):

    def __init__(self, model):
        """
        ModelTrainer for high-level training of Pytorch models
        """
        self.model = model

        # callbacks
        self.history = History(self)
        self._callbacks = [self.history]
        # constraints
        self._constraints = []
        self._has_constraints = False
        self._has_lagrangian_constraints = False
        # regularizers
        self._regularizers = []
        self._has_regularizers = False
        # metrics
        self._metrics = []
        self._has_metrics = False
        # losses
        self._loss_fns = []
        self._has_multiple_loss_fns = False
        # initializers
        self._initializers = []
        self._has_initializers = False
        # transforms
        self._transforms = []
        self._has_transforms = False
        self._has_input_transform = False
        self._has_target_transform = False
        self._has_co_transform = False

        # other properties
        self._stop_training = False

    def forward(self, *input):
        """
        Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError('Subclass must implement this method')

    def summary(self, input_size):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

                params = 0
                if hasattr(module, 'weight'):
                    params += th.prod(th.LongTensor(list(module.weight.size())))
                    if module.weight.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False
                if hasattr(module, 'bias'):
                    params +=  th.prod(th.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = params
                
            if not isinstance(module, nn.Sequential) and \
               not isinstance(module, nn.ModuleList) and \
               not (module == self.model):
                hooks.append(module.register_forward_hook(hook))

        if isinstance(input_size[0], list):
            x = [Variable(th.rand(1,*in_size)) for in_size in input_size]
        else:
            x = Variable(th.rand(1,*input_size))

        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        self.model.apply(register_hook)
        # make a forward pass
        self.model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        return summary

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

    def set_regularizers(self, regularizers):
        if not isinstance(regularizers, (list,tuple)):
            regularizers = [regularizers]
        self._regularizers = regularizers
        self._has_regularizers = True

    def add_regularizer(self, regularizer):
        self._regularizers.append(regularizer)
        self._has_regularizers = True

    def set_constraints(self, constraints):
        if not isinstance(constraints, (list,tuple)):
            constraints = [constraints]
        self._constraints = constraints
        self._has_constraints = True
        if any([c.lagrangian for c in self._constraints]):
            self._has_lagrangian_constraints = True

    def add_constraint(self, constraint):
        self._constraints.append(constraint)
        self._has_constraints = True
        if constraint.lagrangian:
            self._has_lagrangian_constraints = True

    def set_callbacks(self, callbacks):
        self._callbacks += callbacks

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def set_metrics(self, metrics):
        if not isinstance(metrics, (list,tuple)):
            metrics = [metrics]
        metrics = [_validate_metric_input(m) for m in metrics]
        self._has_metrics = True
        self._metrics = metrics

    def add_metric(self, metric):
        self._metrics.append(_validate_metric_input(metric))
        self._has_metrics = True

    def set_initializers(self, initializers):
        if not isinstance(initializers, (list,tuple)):
            initializers = [initializers]
        initializers = [_validate_initializer_input(it) for it in initializers]
        self._has_initializers = True
        self._initializers = initializers

    def add_initializer(self, initializer):
        self._initializers.append(_validate_initializer_input(initializer))
        self._has_initializers = True

    def set_transforms(self, transforms):
        if not isinstance(transforms, (list,tuple)):
            transforms = [transforms, None, None]
        if len(transforms) == 1:
            transforms = [transforms, None, None]
        elif len(transforms) == 2:
            transforms = [transforms, transforms, None]

        if transforms[0] is not None:
            self._has_input_transform = True
        else:
            self._has_input_transform = False
        if transforms[1] is not None:
            self._has_target_transform = True
        else:
            self._has_target_transform = False
        if transforms[2] is not None:
            self._has_co_transform = True
        else:
            self._has_co_transform = False

        self._has_transforms = True
        self._transforms = transforms

    def compile(self,
                optimizer,
                loss,
                regularizers=None,
                initializers=None,
                callbacks=None,
                constraints=None,
                metrics=None,
                transforms=None,
                **kwargs):
        opt_kwargs = {k.split('optimizer_')[1]:v for k,v in kwargs.items() if 'optimizer_' in k}
        self.set_optimizer(optimizer, **opt_kwargs)
        self.set_loss(loss)
        if regularizers is not None:
            self.set_regularizers(regularizers)
        if initializers is not None:
            self.set_initializers(initializers)
        if callbacks is not None:
            self.set_callbacks(callbacks)
        if constraints is not None:
            self.set_constraints(constraints)
        if metrics is not None:
            self.set_metrics(metrics)
        if transforms is not None:
            self.set_transforms(transforms)

    def fit(self,
            inputs,
            targets=None,
            val_data=None,
            nb_epoch=100,
            batch_size=32,
            shuffle=False,
            cuda_devices=[],
            verbose=1):
        # convert inputs to a list if not already
        if not isinstance(inputs, (list,tuple)):
            inputs = [inputs]

        if cuda_devices and th.cuda.is_available():
            # turn on the cudnn autotuner that selects efficient algorithms
            cudnn.benchmark = True

        # determine whether targets were given
        # and convert targets to list if not already
        if targets is None:
            has_target = False
        else:
            has_target = True
            if not isinstance(targets, (list,tuple)):
                targets = [targets]
            nb_targets = len(targets)

        # store whether validation data was given
        if val_data is None:
            has_validation_data = False
        else:
            has_validation_data = True      

        # create regularizers
        if hasattr(self.model, 'regularizers'):
            for reg in self.model.regularizers:
                self.add_regularizer(reg)
        if self._has_regularizers:
            regularizers = RegularizerModule(self._regularizers)

        # create constraints
        if hasattr(self.model, 'constraints'):
            for constraint in self.model.constraints:
                self.add_constraint(constraint)
        if self._has_constraints:
            constraints = ConstraintModule(self._constraints)
            constraints.set_model(self.model)

        # create metrics
        if hasattr(self.model, 'metrics'):
            for metric in self.model.metrics:
                self.add_metric(metric)
        if self._has_metrics:
            metrics = MetricsModule(self._metrics)

        # create initializers
        if hasattr(self.model, 'initializers'):
            for initializer in self.model.initializers:
                self.add_initializer(initializer)
        if self._has_initializers:
            initializers = InitializerModule(self._initializers)
            initializers(self.model)

        # enter context-manager for progress bar
        with TQDM() as pbar:
            # create callbacks
            progressbar = []
            # add progress bar if necessary
            if verbose > 0:
                progressbar = [pbar]
            callbacks = CallbackModule(self._callbacks + progressbar)
            callbacks.set_model(self)

            train_begin_logs = {
                'start_time': _get_current_time(),
                'has_validation_data': has_validation_data
            }
            callbacks.on_train_begin(logs=train_begin_logs)

            # calculate total number of batches
            nb_batches = int(math.ceil(len(inputs[0]) / batch_size))

            # loop through each epoch
            for epoch_idx in range(nb_epoch):
                epoch_logs = {
                    'nb_batches': nb_batches,
                    'nb_epoch': nb_epoch,
                    'has_validation_data': has_validation_data
                }
                callbacks.on_epoch_begin(epoch_idx, epoch_logs)

                # shuffle inputs and targets if necessary
                if shuffle:
                    rand_idx = th.randperm(len(inputs[0]))
                    inputs = [ins[rand_idx] for ins in inputs]
                    targets = [tars[rand_idx] for tars in targets]

                # loop through each batch
                for batch_idx in range(nb_batches):
                    batch_logs = {'batch_idx': batch_idx}  
                    callbacks.on_batch_begin(batch_idx, batch_logs) 

                    # grab an input batch and a target batch if necessary
                    input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size]) for x in inputs]
                    if has_target:
                        target_batch = [Variable(y[batch_idx*batch_size:(batch_idx+1)*batch_size]) for y in targets]

                    # move batch to GPU if necessary (note: all tensors must start on the same GPU (hence we select first GPU in the list))
                    if cuda_devices:
                        input_batch = [ins.cuda(device_id=cuda_devices[0]) for ins in input_batch]
                        if has_target:
                            target_batch = [targs.cuda(device_id=cuda_devices[0]) for targs in target_batch]

                    # apply input, target, and input+target transforms if necessary
                    if self._has_input_transform:
                        input_batch = [self._transforms[0](ins) for ins in input_batch]
                    if self._has_target_transform:
                        target_batch = [self._transforms[1](tars) for tars in target_batch]
                    if self._has_co_transform:
                        input_batch, target_batch = zip(*[self._transforms[2](ins, tars) for ins, tars in zip(input_batch, target_batch)])
                    
                    batch_logs['batch_samples'] = len(input_batch[0])

                    # zero grads and forward pass
                    self._optimizer.zero_grad()
                    outputs = self.model(*input_batch)

                    # apply multiple loss functions if necessary
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
                        # multiple outputs, but they all go into one loss functions
                        if len(outputs) == _nb_function_args(self._loss_fns[0]):
                            loss = self._loss_fns[0](*outputs)
                        # multiple outputs, each with their own loss function
                        else:
                            loss = self._loss_fns[0](outputs[0])
                            for loss_idx in range(1,len(outputs)):
                                if self._has_multiple_loss_fns:
                                    loss += self._loss_fns[loss_idx](outputs[loss_idx])
                                else:
                                    loss += self._loss_fns[0](outputs[loss_idx])
                        
                    # add regularizers to loss if necessary
                    if self._has_regularizers:
                        regularizer_loss = regularizers(self.model)
                        loss += regularizer_loss
                        batch_logs['regularizer_loss'] = regularizer_loss.data[0]

                    # add lagrangian constraints to loss if necessary
                    if self._has_lagrangian_constraints:
                        constraint_loss = constraints(self.model)
                        loss += constraint_loss
                        batch_logs['constraint_loss'] = constraint_loss.data[0]

                    batch_logs['loss'] = loss.data[0]

                    # calculate custom/special batch metrics if necessary
                    if self._has_metrics:
                        metric_logs = metrics(outputs[0], target_batch[0])
                        batch_logs.update(metric_logs)

                    # backward pass and optimizer step
                    loss.backward()
                    self._optimizer.step()

                    callbacks.on_batch_end(batch_idx, batch_logs)

                    # apply explicit constraints if necessary
                    if self._has_constraints:
                        constraints.on_batch_end(batch_idx)

                # validation evaluation if necessary
                if has_validation_data:
                    val_loss = self.evaluate(*val_data, 
                                             batch_size=batch_size,
                                             cuda_device=cuda_devices[0])
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

                # apply Epoch-level constraints if necessary
                if self._has_constraints:
                    constraints.on_epoch_end(epoch_idx)
                # reset all metric counters
                if self._has_metrics:
                    metrics.reset()
                # exit the training loop if necessary (e.g. EarlyStopping)
                if self._stop_training:
                    break

        train_logs = {
            'final_loss': self.history.losses[-1],
            'best_loss': min(self.history.losses),
            'end_time': _get_current_time()
        }
        if has_validation_data:
            train_logs['final_val_loss'] = self.history.val_losses[-1]
            train_logs['best_val_loss'] = min(self.history.val_losses)

        callbacks.on_train_end(logs=train_logs)

    def fit_loader(self, 
                   loader, 
                   val_loader=None, 
                   nb_epoch=100,
                   cuda_devices=[],
                   metrics=None,
                   verbose=1):

        if cuda_devices and th.cuda.is_available():
            # turn on the cudnn autotuner that selects efficient algorithms
            cudnn.benchmark = True

        # store whether validation data was given
        if val_loader is None:
            has_validation_data = False
        else:
            has_validation_data = True      

        # create regularizers
        if hasattr(self.model, 'regularizers'):
            for reg in self.model.regularizers:
                self.add_regularizer(reg)
        if self._has_regularizers:
            regularizers = RegularizerModule(self._regularizers)

        # create constraints
        if hasattr(self.model, 'constraints'):
            for constraint in self.model.constraints:
                self.add_constraint(constraint)
        if self._has_constraints:
            constraints = ConstraintModule(self._constraints)
            constraints.set_model(self.model)

        # create metrics
        if hasattr(self.model, 'metrics'):
            for metric in self.model.metrics:
                self.add_metric(metric)
        if self._has_metrics:
            metrics = MetricsModule(self._metrics)

        # create initializers
        if hasattr(self.model, 'initializers'):
            for initializer in self.model.initializers:
                self.add_initializer(initializer)
        if self._has_initializers:
            initializers = InitializerModule(self._initializers)
            initializers(self.model)

        # Handle multiple GPUs. Single gpu gets normal treatment while multi-GPU must be wrapped in DataParallel
        if len(cuda_devices) == 1:
            self.model.cuda(device_id=cuda_devices[0])
        elif len(cuda_devices) > 1:
            self.model = th.nn.DataParallel(self.model, device_ids=cuda_devices)

        # enter context-manager for progress bar
        with TQDM() as pbar:
            # create callbacks
            progressbar = []
            # add progress bar if necessary
            if verbose > 0:
                progressbar = [pbar]
            callbacks = CallbackModule(self._callbacks + progressbar)
            callbacks.set_model(self)

            train_begin_logs = {
                'start_time': _get_current_time(),
                'has_validation_data': has_validation_data
            }
            callbacks.on_train_begin(logs=train_begin_logs)

            # calculate total number of batches
            nb_batches = int(math.ceil(len(loader.dataset) / loader.batch_size))

            # loop through each epoch
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

                    if cuda_devices:
                        input_batch = [ins.cuda(device_id=cuda_devices[0]) for ins in input_batch]
                        if has_target:
                            target_batch = [targs.cuda(device_id=cuda_devices[0]) for targs in target_batch]

                    # apply input, target, and input+target transforms if necessary
                    if self._has_input_transform:
                        input_batch = [self._transforms[0](ins) for ins in input_batch]
                    if self._has_target_transform:
                        target_batch = [self._transforms[1](tars) for tars in target_batch]
                    if self._has_co_transform:
                        input_batch, target_batch = zip(*[self._transforms[2](ins, tars) for ins, tars in zip(input_batch, target_batch)])
                    
                    batch_logs['batch_samples'] = len(input_batch[0])

                    # zero grads and forward pass
                    self._optimizer.zero_grad()
                    outputs = self.model(*input_batch)

                    # apply multiple loss functions if necessary
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
                        # multiple outputs, but they all go into one loss functions
                        if len(outputs) == _nb_function_args(self._loss_fns[0]):
                            loss = self._loss_fns[0](*outputs)
                        # multiple outputs, each with their own loss function
                        else:
                            loss = self._loss_fns[0](outputs[0])
                            for loss_idx in range(1,len(outputs)):
                                if self._has_multiple_loss_fns:
                                    loss += self._loss_fns[loss_idx](outputs[loss_idx])
                                else:
                                    loss += self._loss_fns[0](outputs[loss_idx])
                        
                    # add regularizers to loss if necessary
                    if self._has_regularizers:
                        regularizer_loss = regularizers(self.model)
                        loss += regularizer_loss
                        batch_logs['regularizer_loss'] = regularizer_loss.data[0]

                    # add lagrangian constraints to loss if necessary
                    if self._has_lagrangian_constraints:
                        constraint_loss = constraints(self.model)
                        loss += constraint_loss
                        batch_logs['constraint_loss'] = constraint_loss.data[0]

                    batch_logs['loss'] = loss.data[0]

                    # calculate custom/special batch metrics if necessary
                    if self._has_metrics:
                        metric_logs = metrics(outputs[0], target_batch[0])
                        batch_logs.update(metric_logs)

                    # backward pass and optimizer step
                    loss.backward()
                    self._optimizer.step()

                    callbacks.on_batch_end(batch_idx, batch_logs)

                    # apply explicit constraints if necessary
                    if self._has_constraints:
                        constraints.on_batch_end(batch_idx)

                if has_validation_data:
                    val_loss = self.evaluate_loader(val_loader,
                                                    cuda_device=cuda_devices[0])
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

                # apply Epoch-level constraints if necessary
                if self._has_constraints:
                    constraints.on_epoch_end(epoch_idx)
                # reset all metric counters
                if self._has_metrics:
                    metrics.reset()
                # exit the training loop if necessary (e.g. EarlyStopping)
                if self._stop_training:
                    break

        train_logs = {
            'final_loss': self.history.losses[-1],
            'best_loss': min(self.history.losses),
            'end_time': _get_current_time()
        }
        if has_validation_data:
            train_logs['final_val_loss'] = self.history.val_losses[-1]
            train_logs['best_val_loss'] = min(self.history.val_losses)

        callbacks.on_train_end(logs=train_logs)

    def train_on_batch(self, 
                       inputs, 
                       targets=None,
                       regularizers=None,
                       constraints=None,
                       callbacks=None,
                       cuda_devices=[]):
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

        if cuda_devices and th.cuda.is_available():
            # turn on the cudnn autotuner that selects efficient algorithms
            cudnn.benchmark = True
            input_batch = [ins.cuda(device_id=cuda_devices[0]) for ins in input_batch]
            if has_target:
                target_batch = [targs.cuda(device_id=cuda_devices[0]) for targs in target_batch]
         
        ## ZERO GRAD AND FORWARD PASS
        self._optimizer.zero_grad()
        outputs = self.model(*input_batch)

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
            regularizer_loss = regularizers(self.model)
            loss += regularizer_loss

        if self._has_constraints and constraints.has_lagrangian:
            constraint_loss = constraints(self.model)
            loss += constraint_loss

        # BACKWARD PASS AND OPTIMIZER STEP
        loss.backward()
        self._optimizer.step()

    def predict(self, 
                inputs, 
                batch_size=32,
                cuda_devices=[],
                verbose=1):
        if not isinstance(inputs, (list,tuple)):
            inputs = [inputs]

        if cuda_devices and th.cuda.is_available():
            # turn on the cudnn autotuner that selects efficient algorithms
            cudnn.benchmark = True

        nb_batches = int(math.ceil(len(inputs[0]) / batch_size))
        prediction_list = []
        for batch_idx in range(nb_batches):
            input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size], volatile=True) for x in inputs]

            if cuda_devices:
                input_batch = [ins.cuda(device_id=cuda_devices[0]) for ins in input_batch]

            prediction_list.append(self.model(*input_batch))
        
        # concatenate all outputs of the same type together (when there are multiple outputs)
        if len(prediction_list) > 0 and isinstance(prediction_list[0], (tuple,list)):
            nb_out = len(prediction_list[0])
            out_list = []
            for out_i in range(nb_out):
                precdiction_out_i = [prediction[out_i] for prediction in prediction_list]
                out_list.append(th.cat(precdiction_out_i, 0))
            return out_list
            
        return th.cat(prediction_list,0)

    def predict_loader(self,
                       loader,
                       cuda_devices=[],
                       verbose=1):
        prediction_list = []

        if cuda_devices and th.cuda.is_available():
            # turn on the cudnn autotuner that selects efficient algorithms
            cudnn.benchmark = True

        for batch_idx, batch_data in enumerate(loader):
            if not isinstance(batch_data, (tuple,list)):
                batch_data = [batch_data]
            input_batch = batch_data[0]
            if not isinstance(input_batch, (list,tuple)):
                input_batch = [input_batch]
            input_batch = [Variable(ins, volatile=True) for ins in input_batch]
            if cuda_devices:
                input_batch = [ins.cuda(device_id=cuda_devices[0]) for ins in input_batch]

            prediction_list.append(self.model(*input_batch))
            
        # concatenate all outputs of the same type together (when there are multiple outputs)
        if len(prediction_list) > 0 and isinstance(prediction_list[0], (tuple,list)):
            nb_out = len(prediction_list[0])
            out_list = []
            for out_i in range(nb_out):
                precdiction_out_i = [prediction[out_i] for prediction in prediction_list]
                out_list.append(th.cat(precdiction_out_i, 0))
            return out_list
            
        return th.cat(prediction_list,0)

    def predict_on_batch(self, 
                         inputs, 
                         cuda_devices=[]):
        if not isinstance(inputs, (list,tuple)):
            inputs = [inputs]

        if cuda_devices and th.cuda.is_available():
            # turn on the cudnn autotuner that selects efficient algorithms
            cudnn.benchmark = True
            inputs = [ins.cuda(device_id=cuda_devices[0]) for ins in inputs]

        preds = self.model(*inputs)
        return preds

    def evaluate(self, 
                 inputs, 
                 targets=None, 
                 batch_size=32,
                 cuda_devices=[],
                 verbose=1):
        # put model in evaluation mode
        self.model.eval()
        if not isinstance(inputs, (list,tuple)):
            inputs = [inputs]

        if cuda_devices and th.cuda.is_available():
            # turn on the cudnn autotuner that selects efficient algorithms
            cudnn.benchmark = True

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
        nb_batches = int(math.ceil(len(inputs[0]) / batch_size))
        for batch_idx in range(nb_batches):
            input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size], volatile=True) for x in inputs]
            if has_target:
                target_batch = [Variable(y[batch_idx*batch_size:(batch_idx+1)*batch_size], volatile=True) for y in targets]

            if cuda_devices:
                input_batch = [ins.cuda(device_id=cuda_devices[0]) for ins in input_batch]
                if has_target:
                    target_batch = [targs.cuda(device_id=cuda_devices[0]) for targs in target_batch]

            outputs = self.model(*input_batch)
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
        self.model.train()
        if self._has_metrics:
            return total_loss / float(total_samples), {k.split('_metric')[0]:v for k,v in metric_logs.items()}
        else:
            return total_loss / float(total_samples)

    def evaluate_loader(self, 
                        loader, 
                        cuda_devices=[]):

        if cuda_devices and th.cuda.is_available():
            # turn on the cudnn autotuner that selects efficient algorithms
            cudnn.benchmark = True

        self.model.eval()
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
            input_batch = [Variable(ins, volatile=True) for ins in input_batch]
            if has_target:
                if not isinstance(target_batch, (list,tuple)):
                    target_batch = [target_batch]
                target_batch = [Variable(targs, volatile=True) for targs in target_batch]
                nb_targets = len(target_batch)

            if cuda_devices:
                input_batch = [ins.cuda(device_id=cuda_devices[0]) for ins in input_batch]
                if has_target:
                    target_batch = [targs.cuda(device_id=cuda_devices[0]) for targs in target_batch]

            outputs = self.model(*input_batch)

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

        self.model.train()
        if self._has_metrics:
            return total_loss / float(total_samples), metric_logs
        else:
            return total_loss / float(total_samples)

    def evaluate_on_batch(self, 
                          inputs, 
                          targets, 
                          cuda_devices=[]):
        self.model.eval()
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

        input_batch = [Variable(x, volatile=True) for x in inputs]
        if has_target:
            target_batch = [Variable(y, volatile=True) for y in targets]

        if cuda_devices and th.cuda.is_available():
            # turn on the cudnn autotuner that selects efficient algorithms
            cudnn.benchmark = True
            input_batch = [ins.cuda(device_id=cuda_devices[0]) for ins in input_batch]
            if has_target:
                target_batch = [targs.cuda(device_id=cuda_devices[0]) for targs in target_batch]

        outputs = self.model(*input_batch)
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
        self.model.train()
        return loss.data[0]

    def save_state_dict(self, file):
        """
        Save a model parameters to disk
        """
        # model parameters -> ordered dict
        state_dict = self.model.state_dict()
        th.save(state_dict, file)







