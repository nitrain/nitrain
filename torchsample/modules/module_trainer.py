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
                     _standardize_user_data, _parse_num_inputs_and_targets,
                     _is_iterable)

from ..callbacks import CallbackContainer, History, TQDM
from ..regularizers import RegularizerContainer, RegularizerCallback
from ..initializers import InitializerContainer
from ..constraints import ConstraintContainer, ConstraintCallback
from ..metrics import MetricContainer, MetricCallback


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
        self._loss = None
        self._loss_fn = None

        # other properties
        self._in_train_loop = False
        self._stop_training = False

    def set_loss(self, loss):
        self._loss = loss
        if _is_iterable(loss):
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

    def set_callbacks(self, callbacks):
        if not _is_iterable(callbacks):
            callbacks = [callbacks]
        self._callbacks = [self.history] + callbacks

    def set_regularizers(self, regularizers):
        regularizers = [regularizers] if not _is_iterable(regularizers) else regularizers
        self._regularizers = regularizers
        self._has_regularizers = True

    def set_initializers(self, initializers):
        initializers = [initializers] if not _is_iterable(initializers) else initializers
        initializers = [_validate_initializer_input(it) for it in initializers]
        self._has_initializers = True
        self._initializers = initializers

    def set_constraints(self, constraints):
        constraints = [constraints] if not _is_iterable(constraints) else constraints
        self._has_constraints = True
        self._constraints = constraints

    def set_metrics(self, metrics):
        metrics = [metrics] if not _is_iterable(metrics) else metrics
        metrics = [_validate_metric_input(m) for m in metrics]
        self._has_metrics = True
        self._metrics = metrics

    def compile(self,
                optimizer,
                loss,
                regularizers=None,
                initializers=None,
                constraints=None,
                metrics=None,
                callbacks=None,
                transforms=None):
        self.set_optimizer(optimizer)
        self.set_loss(loss)

        if regularizers is not None:
            self.set_regularizers(regularizers)
            self.regularizer_container = RegularizerContainer(self._regularizers)
            self.regularizer_container.register_forward_hooks(self.model)
        else:
            self._has_regularizers = False

        if callbacks is not None:
            self.set_callbacks(callbacks)

        if initializers is not None:
            self.set_initializers(initializers)
            self.initializer_container = InitializerContainer(self._initializers)
            # actually initialize the model
            self.initializer_container.apply(self.model)

        if constraints is not None:
            self.set_constraints(constraints)
            self.constraint_container = ConstraintContainer(self._constraints)
            self.constraint_container.register_constraints(self.model)
        else:
            self._has_constraints = False

        if metrics is not None:
            self.set_metrics(metrics)
            self.metric_container = MetricContainer(self._metrics)
        else:
            self._has_metrics = False

    def fit(self,
            inputs,
            targets=None,
            val_data=None,
            num_epoch=100,
            batch_size=32,
            shuffle=False,
            cuda_device=-1,
            verbose=1):
        """
        Fit a model on in-memory tensors using ModuleTrainer
        """
        num_inputs, num_targets = _parse_num_inputs_and_targets(inputs, targets)
        has_val_data = val_data is not None
        if has_val_data:
            if num_targets == 0:
                val_data = (val_data, None)
            if len(val_data) != 2:
                raise ValueError('val_data must be a 2-tuple')
            num_val_inputs, num_val_targets = _parse_num_inputs_and_targets(val_data[0], val_data[1])
            if (num_inputs != num_val_inputs) or (num_targets != num_val_targets):
                raise ValueError('num_inputs != num_val_inputs or num_targets != num_val_targets')
            val_inputs, val_targets = val_data[0], val_data[1]

        fitter = _get_helper(self, num_inputs, num_targets)

        if cuda_device > -1:
            inputs, targets = fitter.move_to_cuda(cuda_device, inputs, targets)

        len_inputs = len(inputs) if not _is_iterable(inputs) else len(inputs[0])
        num_batches = int(math.ceil(len_inputs / batch_size))

        with TQDM() as pbar:
            tmp_callbacks = []
            if verbose > 0:
                tmp_callbacks.append(pbar)
            if self._has_regularizers:
                tmp_callbacks.append(RegularizerCallback(self.regularizer_container))
            if self._has_constraints:
                tmp_callbacks.append(ConstraintCallback(self.constraint_container))
            if self._has_metrics:
                tmp_callbacks.append(MetricCallback(self.metric_container, helper=fitter))

            callback_container = CallbackContainer(self._callbacks+tmp_callbacks)
            callback_container.set_trainer(self)
            callback_container.on_train_begin({'batch_size': batch_size,
                                               'num_batches': num_batches,
                                               'num_epoch': num_epoch,
                                               'has_val_data': has_val_data,
                                               'has_regularizers': self._has_regularizers,
                                               'has_metrics': self._has_metrics})

            for epoch_idx in range(num_epoch):
                epoch_logs = {}
                callback_container.on_epoch_begin(epoch_idx, epoch_logs)

                if shuffle:
                    inputs, targets = fitter.shuffle_arrays(inputs, targets)

                for batch_idx in range(num_batches):
                    batch_logs = {}
                    callback_container.on_batch_begin(batch_idx, batch_logs)

                    input_batch, target_batch = fitter.grab_batch(batch_idx, batch_size, inputs, targets)

                    self._optimizer.zero_grad()
                    output_batch = fitter.forward_pass(self.model, input_batch)
                    loss = fitter.calculate_loss(self._loss_fn, output_batch, target_batch)
                    loss.backward()

                    if self._has_regularizers:
                        regularizer_loss = self.regularizer_container.get_value()
                        loss += regularizer_loss
                        batch_logs['reg_loss'] = regularizer_loss.data[0]

                    self._optimizer.step()

                    if self._has_metrics:
                        metrics_logs = self.metric_container(output_batch, target_batch)
                        batch_logs.update(metrics_logs)

                    batch_logs['loss'] = loss.data[0]
                    callback_container.on_batch_end(batch_idx, batch_logs)

                if has_val_data:
                    self._in_train_loop = True
                    val_epoch_logs = self.evaluate(val_inputs,
                                                   val_targets,
                                                   batch_size=batch_size,
                                                   cuda_device=cuda_device,
                                                   verbose=verbose)
                    self._in_train_loop = False
                    self.history.batch_metrics.update(val_epoch_logs)

                callback_container.on_epoch_end(epoch_idx, epoch_logs)

                if self._stop_training:
                    break

    def predict(self,
                inputs,
                batch_size=32,
                cuda_device=-1,
                verbose=1):
        if _is_iterable(inputs):
            num_inputs = len(inputs)
            len_inputs = len(inputs[0])
        else:
            num_inputs = 1
            len_inputs = len(inputs)
        predictor = _get_helper(self, num_inputs, num_targets=0)

        if cuda_device >= 0:
            inputs = predictor.move_to_cuda(cuda_device, inputs)
            self.model.cuda(cuda_device)

        num_batches = int(math.ceil(len_inputs / batch_size))
        
        for batch_idx in range(num_batches):
            input_batch, _ = predictor.grab_batch(batch_idx, batch_size, inputs)
            output_batch = predictor.forward_pass(self.model, input_batch)

            if batch_idx == 0:
                len_outputs = 1 if not _is_iterable(output_batch) else len(output_batch)
                prediction_lists = [[] for _ in range(len_outputs)]

            if len_outputs == 1:
                prediction_lists[0].append(output_batch)
            else:
                for out_idx in range(len_outputs):
                    prediction_lists[out_idx].append(output_batch[out_idx])
            
        final_pred_list = [th.cat(pred_list,0) for pred_list in prediction_lists]
        return final_pred_list if len_outputs > 1 else final_pred_list[0]

    def evaluate(self,
                 inputs,
                 targets=None,
                 batch_size=32,
                 cuda_device=-1,
                 verbose=1):
        num_inputs, num_targets = _parse_num_inputs_and_targets(inputs, targets)
        evaluator = _get_helper(self, num_inputs, num_targets)

        if cuda_device > -1:
            inputs, targets = evaluator.move_to_cuda(cuda_device, inputs, targets)

        len_inputs = len(inputs) if not _is_iterable(inputs) else len(inputs[0])
        num_batches = int(math.ceil(len_inputs / batch_size))

        eval_logs= {
            'val_loss': 0.
        }
        samples_seen = 0
        for batch_idx in range(num_batches):
            input_batch, target_batch = evaluator.grab_batch(batch_idx, batch_size, inputs, targets)

            self._optimizer.zero_grad()
            output_batch = evaluator.forward_pass(self.model, input_batch)
            loss = evaluator.calculate_loss(self._loss_fn, output_batch, target_batch)
            
            samples_seen += batch_size
            eval_logs['val_loss'] = (samples_seen*eval_logs['val_loss'] + loss.data[0]*batch_size) / (samples_seen+batch_size)

        if self._in_train_loop:
            return eval_logs
        else:
            return eval_logs['val_loss']


def _get_helper(trainer, num_inputs, num_targets):
    if (num_inputs == 1) and (num_targets == 1):
        fitter = SingleInput_SingleTarget_Helper()

    elif (num_inputs == 1) and (num_targets > 1):
        # use same loss function for all targets if multiple loss fns not explicitly given
        if not _is_iterable(trainer._loss_fn):
            trainer._loss_fn = [trainer._loss_fn] * num_targets
        else:
            if len(trainer._loss_fn) != num_targets:
                raise ValueError('must give one loss function for every input if you give multiple')
        fitter = SingleInput_MultiTarget_Helper()

    elif (num_inputs == 1) and (num_targets == 0):
        fitter = SingleInput_NoTarget_Helper()

    elif (num_inputs > 1) and (num_targets == 1):
        fitter = MultiInput_SingleTarget_Helper()

    elif (num_inputs > 1) and (num_targets > 1):
        # use same loss function for all targets if multiple loss fns not explicitly given
        if not _is_iterable(trainer._loss_fn):
            trainer._loss_fn = [trainer._loss_fn] * num_targets
        else:
            if len(trainer._loss_fn) != num_targets:
                raise ValueError('must give one loss function for every input if you give multiple')
        fitter = MultiInput_MultiTarget_Helper()

    elif (num_inputs > 1) and (num_targets == 0):
        fitter = MultiInput_NoTarget_Helper()

    return fitter


class SingleInput_SingleTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = inputs.cuda(cuda_device)
        targets = targets.cuda(cuda_device)
        return inputs, targets
    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        targets = targets[rand_indices]
        return inputs, targets
    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = Variable(inputs[batch_idx*batch_size:(batch_idx+1)*batch_size])
        target_batch = Variable(targets[batch_idx*batch_size:(batch_idx+1)*batch_size])
        return input_batch, target_batch
    def forward_pass(self, model, input_batch):
        return model(input_batch)
    def calculate_loss(self, loss_fn, output_batch, target_batch):
        return loss_fn(output_batch, target_batch)


class SingleInput_MultiTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = inputs.cuda(cuda_device)
        targets = [target_.cuda(cuda_device) for target_ in targets]
        return inputs, targets
    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        targets = [target_[rand_indices] for target_ in targets]
        return inputs, targets
    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = Variable(inputs[batch_idx*batch_size:(batch_idx+1)*batch_size])
        target_batch = [Variable(target_[batch_idx*batch_size:(batch_idx+1)*batch_size])
                        for target_ in targets]
        return input_batch, target_batch
    def forward_pass(self, model, input_batch):
        return model(input_batch)
    def calculate_loss(self, loss_fn, output_batch, target_batch):
        return sum([loss_fn[idx](output_batch[idx], target_batch[idx]) 
                    for idx in range(len(output_batch))])


class MultiInput_SingleTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = [input_.cuda(cuda_device) for input_ in inputs] 
        targets = targets.cuda(cuda_device)
        return inputs, targets
    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        targets = targets[rand_indices]
        return inputs, targets
    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = [Variable(input_[batch_idx*batch_size:(batch_idx+1)*batch_size])
                       for input_ in inputs]
        target_batch = Variable(targets[batch_idx*batch_size:(batch_idx+1)*batch_size])
        return input_batch, target_batch
    def forward_pass(self, model, input_batch):
        return model(*input_batch)
    def calculate_loss(self, loss_fn, output_batch, target_batch):
        return loss_fn(output_batch, target_batch)


class MultiInput_MultiTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets):
        inputs = [input_.cuda(cuda_device) for input_ in inputs] 
        targets = [target_.cuda(cuda_device) for target_ in targets]
        return inputs, targets
    def shuffle_arrays(self, inputs, targets):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        targets = [input_[rand_indices] for input_ in inputs]
        return inputs, targets
    def grab_batch(self, batch_idx, batch_size, inputs, targets):
        input_batch = [Variable(input_[batch_idx*batch_size:(batch_idx+1)*batch_size])
                       for input_ in inputs]
        target_batch = [Variable(target_[batch_idx*batch_size:(batch_idx+1)*batch_size])
                       for target_ in targets]
        return input_batch, target_batch
    def forward_pass(self, model, input_batch):
        return model(*input_batch)
    def calculate_loss(self, loss_fn, output_batch, target_batch):
        return sum([loss_fn[idx](output_batch[idx], target_batch[idx]) 
                    for idx in range(len(output_batch))])


class SingleInput_NoTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets=None):
        inputs = inputs.cuda(cuda_device)
        return inputs, None
    def shuffle_arrays(self, inputs, targets=None):
        rand_indices = th.randperm(len(inputs))
        inputs = inputs[rand_indices]
        return inputs, None
    def grab_batch(self, batch_idx, batch_size, inputs, targets=None):
        input_batch = Variable(inputs[batch_idx*batch_size:(batch_idx+1)*batch_size])
        return input_batch, None
    def forward_pass(self, model, input_batch):
        return model(input_batch)
    def calculate_loss(self, loss_fn, output_batch, target_batch=None):
        return loss_fn(output_batch)


class MultiInput_NoTarget_Helper(object):
    def move_to_cuda(self, cuda_device, inputs, targets=None):
        inputs = [input_.cuda(cuda_device) for input_ in inputs]
        return inputs, None
    def shuffle_arrays(self, inputs, targets=None):
        rand_indices = th.randperm(len(inputs))
        inputs = [input_[rand_indices] for input_ in inputs]
        return inputs, None
    def grab_batch(self, batch_idx, batch_size, inputs, targets=None):
        input_batch = [Variable(input_[batch_idx*batch_size:(batch_idx+1)*batch_size])
                       for input_ in inputs]
        return input_batch, None
    def forward_pass(self, model, input_batch):
        return model(*input_batch)
    def calculate_loss(self, loss_fn, output_batch, target_batch=None):
        return loss_fn(output_batch)
