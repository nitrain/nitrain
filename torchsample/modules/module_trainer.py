"""
ModuleTrainer for high level training on Pytorch models
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings
import functools
import math
from collections import OrderedDict

import torch as th
import torch.nn as nn
from torch.autograd import Variable

# local imports
from ._utils import (_validate_loss_input, _validate_metric_input,
                     _validate_optimizer_input, _validate_initializer_input,
                     _standardize_user_data, _parse_num_inputs_and_targets,
                     _is_tuple_or_list, _parse_num_inputs_and_targets_from_loader,
                     _add_regularizer_to_loss_fn)

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
        self._callbacks = []

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
        if _is_tuple_or_list(loss):
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
        if not _is_tuple_or_list(callbacks):
            callbacks = [callbacks]
        self._callbacks = [self.history] + callbacks

    def set_regularizers(self, regularizers):
        regularizers = [regularizers] if not _is_tuple_or_list(regularizers) else regularizers
        self._regularizers = regularizers
        self._has_regularizers = True

    def set_initializers(self, initializers):
        initializers = [initializers] if not _is_tuple_or_list(initializers) else initializers
        initializers = [_validate_initializer_input(it) for it in initializers]
        self._initializers = initializers

    def set_constraints(self, constraints):
        constraints = [constraints] if not _is_tuple_or_list(constraints) else constraints
        self._has_constraints = True
        self._constraints = constraints

    def set_metrics(self, metrics):
        metrics = [metrics] if not _is_tuple_or_list(metrics) else metrics
        metrics = [_validate_metric_input(m) for m in metrics]
        self._has_metrics = True
        self._metrics = metrics

    def set_transforms(self, transforms):
        if not _is_tuple_or_list(transforms):
            transforms = (transforms, lambda x: x, lambda x,y: (x,y))
        if len(transforms) == 1:
            transforms = (transforms, lambda x: x, lambda x,y: (x,y))
        elif len(transforms) == 2:
            transforms = (transforms, transforms, lambda x,y: (x,y))

        self._has_input_transform = transforms[0] is not None
        self._has_target_transform = transforms[1] is not None
        self._has_co_transform = transforms[2] is not None

        self._has_transforms = True
        self._transforms = transforms

    def compile(self,
                optimizer,
                loss,
                callbacks=None,
                regularizers=None,
                initializers=None,
                constraints=None,
                metrics=None,
                transforms=None):
        self.set_optimizer(optimizer)
        self.set_loss(loss)

        if regularizers is not None:
            self.set_regularizers(regularizers)
            self.regularizer_container = RegularizerContainer(self._regularizers)
            self.regularizer_container.register_forward_hooks(self.model)
        else:
            self._has_regularizers = False

        self.history = History(self)
        self._callbacks = [self.history]
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
        
        if transforms is not None:
            self.set_transforms(transforms)
        else:
            self._has_transforms = False

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
        # ----------------------------------------------------------------------
        num_inputs, num_targets = _parse_num_inputs_and_targets(inputs, targets)
        len_inputs = len(inputs) if not _is_tuple_or_list(inputs) else len(inputs[0])
        
        if val_data is not None:
            if num_targets == 0:
                val_data = (val_data, None)
            if len(val_data) != 2:
                raise ValueError('val_data must be a 2-tuple')
            num_val_inputs, num_val_targets = _parse_num_inputs_and_targets(val_data[0], val_data[1])
            if (num_inputs != num_val_inputs) or (num_targets != num_val_targets):
                raise ValueError('num_inputs != num_val_inputs or num_targets != num_val_targets')
            val_inputs, val_targets = val_data
        has_val_data = val_data is not None
        num_batches = int(math.ceil(len_inputs / batch_size))
        # ----------------------------------------------------------------------

        fit_helper = _get_helper(self, num_inputs, num_targets)
        fit_loss_fn = fit_helper.get_partial_loss_fn(self._loss_fn)
        fit_forward_fn = fit_helper.get_partial_forward_fn(self.model)

        with TQDM() as pbar:
            tmp_callbacks = []
            if verbose > 0:
                tmp_callbacks.append(pbar)
            if self._has_regularizers:
                tmp_callbacks.append(RegularizerCallback(self.regularizer_container))
                fit_loss_fn = _add_regularizer_to_loss_fn(fit_loss_fn,
                                                          self.regularizer_container)
            if self._has_constraints:
                tmp_callbacks.append(ConstraintCallback(self.constraint_container))
            if self._has_metrics:
                self.metric_container.set_helper(fit_helper)
                tmp_callbacks.append(MetricCallback(self.metric_container))

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
                    inputs, targets = fit_helper.shuffle_arrays(inputs, targets)

                for batch_idx in range(num_batches):
                    batch_logs = {}
                    callback_container.on_batch_begin(batch_idx, batch_logs)

                    input_batch, target_batch = fit_helper.grab_batch(batch_idx, batch_size, inputs, targets)
                    if cuda_device >= 0:
                        input_batch, target_batch = fit_helper.move_to_cuda(cuda_device, input_batch, target_batch)
                    if self._has_transforms:
                        input_batch, target_batch = fit_helper.apply_transforms(self._transforms, input_batch, target_batch)

                    # ---------------------------------------------
                    self._optimizer.zero_grad()
                    output_batch = fit_forward_fn(input_batch)
                    loss = fit_loss_fn(output_batch, target_batch)
                    loss.backward()
                    self._optimizer.step()
                    # ---------------------------------------------

                    if self._has_regularizers:
                        batch_logs['reg_loss'] = self.regularizer_container.current_value
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

    def fit_loader(self,
                   loader,
                   val_loader=None,
                   num_epoch=100,
                   cuda_device=-1,
                   verbose=1):
        """
        Fit a model on in-memory tensors using ModuleTrainer
        """
        # ----------------------------------------------------------------------
        num_inputs = loader.dataset.num_inputs
        num_targets = loader.dataset.num_targets
        len_inputs = len(loader.dataset)
        batch_size = loader.batch_size

        if val_loader is not None:
            num_val_inputs = val_loader.dataset.num_inputs
            num_val_targets = val_loader.dataset.num_targets
            if (num_inputs != num_val_inputs) or (num_targets != num_val_targets):
                raise ValueError('num_inputs != num_val_inputs or num_targets != num_val_targets')
        has_val_data = val_loader is not None
        num_batches = int(math.ceil(len_inputs / batch_size))
        # ----------------------------------------------------------------------

        fit_helper = _get_helper(self, num_inputs, num_targets)
        fit_loss_fn = fit_helper.get_partial_loss_fn(self._loss_fn)
        fit_forward_fn = fit_helper.get_partial_forward_fn(self.model)

        with TQDM() as pbar:
            tmp_callbacks = []
            if verbose > 0:
                tmp_callbacks.append(pbar)
            if self._has_regularizers:
                tmp_callbacks.append(RegularizerCallback(self.regularizer_container))
                fit_loss_fn = _add_regularizer_to_loss_fn(fit_loss_fn,
                                                            self.regularizer_container)
            if self._has_constraints:
                tmp_callbacks.append(ConstraintCallback(self.constraint_container))
            if self._has_metrics:
                self.metric_container.set_helper(fit_helper)
                tmp_callbacks.append(MetricCallback(self.metric_container))

            callback_container = CallbackContainer(self._callbacks+tmp_callbacks)
            callback_container.set_trainer(self)
            callback_container.on_train_begin({'batch_size': loader.batch_size,
                                               'num_batches': num_batches,
                                               'num_epoch': num_epoch,
                                               'has_val_data': has_val_data,
                                               'has_regularizers': self._has_regularizers,
                                               'has_metrics': self._has_metrics})

            for epoch_idx in range(num_epoch):
                epoch_logs = {}
                callback_container.on_epoch_begin(epoch_idx, epoch_logs)

                loader_iter = iter(loader)
                for batch_idx in range(num_batches):

                    batch_logs = {}
                    callback_container.on_batch_begin(batch_idx, batch_logs)

                    input_batch, target_batch = fit_helper.grab_batch_from_loader(loader_iter)
                    if cuda_device >= 0:
                        input_batch, target_batch = fit_helper.move_to_cuda(cuda_device, input_batch, target_batch)
                    
                    # ---------------------------------------------
                    self._optimizer.zero_grad()
                    output_batch = fit_forward_fn(input_batch)
                    loss = fit_loss_fn(output_batch, target_batch)
                    loss.backward()
                    self._optimizer.step()
                    # ---------------------------------------------

                    if self._has_regularizers:
                        batch_logs['reg_loss'] = self.regularizer_container.current_value
                    if self._has_metrics:
                        metrics_logs = self.metric_container(output_batch, target_batch)
                        batch_logs.update(metrics_logs)

                    batch_logs['loss'] = loss.data[0]
                    callback_container.on_batch_end(batch_idx, batch_logs)

                if has_val_data:
                    self._in_train_loop = True
                    val_epoch_logs = self.evaluate_loader(val_loader,
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
        # --------------------------------------------------------
        num_inputs, _ = _parse_num_inputs_and_targets(inputs, None)
        len_inputs = len(inputs) if not _is_tuple_or_list(inputs) else len(inputs[0])
        num_batches = int(math.ceil(len_inputs / batch_size))
        # --------------------------------------------------------

        predict_helper = _get_helper(self, num_inputs, num_targets=0)
        pred_forward_fn = predict_helper.get_partial_forward_fn(self.model)
        
        for batch_idx in range(num_batches):
            input_batch, _ = predict_helper.grab_batch(batch_idx, batch_size, inputs, None)
            if cuda_device >= 0:
                inputs = predict_helper.move_to_cuda(cuda_device, inputs)
            output_batch = pred_forward_fn(input_batch)

            if batch_idx == 0:
                len_outputs = 1 if not _is_tuple_or_list(output_batch) else len(output_batch)
                prediction_lists = [[] for _ in range(len_outputs)]

            if len_outputs == 1:
                prediction_lists[0].append(output_batch)
            else:
                for out_idx in range(len_outputs):
                    prediction_lists[out_idx].append(output_batch[out_idx])
            
        final_pred_list = [th.cat(pred_list,0) for pred_list in prediction_lists]
        return final_pred_list if len_outputs > 1 else final_pred_list[0]

    def predict_loader(self,
                       loader,
                       cuda_device=-1,
                       verbose=1):
        # --------------------------------------------------------
        num_inputs, num_targets = _parse_num_inputs_and_targets_from_loader(loader)
        batch_size = loader.batch_size
        len_inputs = len(loader.dataset)
        num_batches = int(math.ceil(len_inputs / batch_size))
        # --------------------------------------------------------

        predict_helper = _get_helper(self, num_inputs, num_targets=0)
        pred_forward_fn = predict_helper.get_partial_forward_fn(self.model)
        
        for batch_idx in range(num_batches):
            input_batch, _ = predict_helper.grab_batch_from_loader(loader)
            output_batch = pred_forward_fn(input_batch)

            if batch_idx == 0:
                len_outputs = 1 if not _is_tuple_or_list(output_batch) else len(output_batch)
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
        len_inputs = len(inputs) if not _is_tuple_or_list(inputs) else len(inputs[0])
        num_batches = int(math.ceil(len_inputs / batch_size))

        evaluate_helper = _get_helper(self, num_inputs, num_targets)
        eval_loss_fn = evaluate_helper.get_partial_loss_fn(self._loss_fn)
        eval_forward_fn = evaluate_helper.get_partial_forward_fn(self.model)
        eval_logs= {'val_loss': 0.}

        samples_seen = 0
        for batch_idx in range(num_batches):
            input_batch, target_batch = evaluate_helper.grab_batch(batch_idx, batch_size, inputs, targets)
            if cuda_device >= 0:
                input_batch, target_batch = evaluate_helper.move_to_cuda(cuda_device, input_batch, target_batch)

            self._optimizer.zero_grad()
            output_batch = eval_forward_fn(input_batch)
            loss = eval_loss_fn(output_batch, target_batch)
            
            samples_seen += batch_size
            eval_logs['val_loss'] = (samples_seen*eval_logs['val_loss'] + loss.data[0]*batch_size) / (samples_seen+batch_size)

        if self._in_train_loop:
            return eval_logs
        else:
            return eval_logs['val_loss']

    def evaluate_loader(self,
                        loader,
                        cuda_device=-1,
                        verbose=1):
        num_inputs, num_targets = _parse_num_inputs_and_targets_from_loader(loader)
        batch_size = loader.batch_size
        len_inputs = len(loader.dataset)
        num_batches = int(math.ceil(len_inputs / batch_size))

        evaluate_helper = _get_helper(self, num_inputs, num_targets)
        eval_loss_fn = evaluate_helper.get_partial_loss_fn(self._loss_fn)
        eval_forward_fn = evaluate_helper.get_partial_forward_fn(self.model)
        eval_logs= {'val_loss': 0.}
        loader_iter = iter(loader)

        samples_seen = 0
        for batch_idx in range(num_batches):
            input_batch, target_batch = evaluate_helper.grab_batch_from_loader(loader_iter)
            if cuda_device >= 0:
                input_batch, target_batch = evaluate_helper.move_to_cuda(cuda_device, input_batch, target_batch)

            self._optimizer.zero_grad()
            output_batch = eval_forward_fn(input_batch)
            loss = eval_loss_fn(output_batch, target_batch)
            
            samples_seen += batch_size
            eval_logs['val_loss'] = (samples_seen*eval_logs['val_loss'] + loss.data[0]*batch_size) / (samples_seen+batch_size)

        if self._in_train_loop:
            return eval_logs
        else:
            return eval_logs['val_loss']

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

        # create properties
        summary = OrderedDict()
        hooks = []
        # register forward hooks
        self.model.apply(register_hook)

        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(th.rand(1,*in_size)) for in_size in input_size]
            self.model(*x)
        else:
            x = Variable(th.rand(1,*input_size))
            self.model(x)

        # remove these hooks
        for h in hooks:
            h.remove()

        return summary


def _get_helper(trainer, num_inputs, num_targets):
    if (num_inputs == 1) and (num_targets == 1):
        helper = SingleInput_SingleTarget_Helper()

    elif (num_inputs == 1) and (num_targets > 1):
        # use same loss function for all targets if multiple loss fns not explicitly given
        if not _is_tuple_or_list(trainer._loss_fn):
            trainer._loss_fn = [trainer._loss_fn] * num_targets
        else:
            if len(trainer._loss_fn) != num_targets:
                raise ValueError('must give one loss function for every input if you give multiple')
        helper = SingleInput_MultiTarget_Helper()

    elif (num_inputs == 1) and (num_targets == 0):
        helper = SingleInput_NoTarget_Helper()

    elif (num_inputs > 1) and (num_targets == 1):
        helper = MultiInput_SingleTarget_Helper()

    elif (num_inputs > 1) and (num_targets > 1):
        # use same loss function for all targets if multiple loss fns not explicitly given
        if not _is_tuple_or_list(trainer._loss_fn):
            trainer._loss_fn = [trainer._loss_fn] * num_targets
        else:
            if len(trainer._loss_fn) != num_targets:
                raise ValueError('must give one loss function for every input if you give multiple')
        helper = MultiInput_MultiTarget_Helper()

    elif (num_inputs > 1) and (num_targets == 0):
        helper = MultiInput_NoTarget_Helper()

    return helper


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
    def grab_batch_from_loader(self, loader_iter):
        input_batch, target_batch = next(loader_iter)
        return Variable(input_batch), Variable(target_batch)
    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = tforms[0](input_batch)
        target_batch = tforms[1](target_batch)
        input_batch, target_batch = tforms[2](input_batch, target_batch)
        return input_batch, target_batch
    def forward_pass(self, input_batch, model):
        return model(input_batch)
    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)
    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch, target_batch)
    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)
        #def new_loss_fn(output_batch, target_batch):
        #    return self.calculate_loss(output_batch, target_batch, loss_fn)
        #return new_loss_fn


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
    def grab_batch_from_loader(self, loader_iter):
        input_batch, target_batch = next(loader_iter)
        return Variable(input_batch), [Variable(target_) for target_ in target_batch]
    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = tforms[0](input_batch)
        target_batch = [tforms[1](target_) for target_ in target_batch]
        return input_batch, target_batch
    def forward_pass(self, input_batch, model):
        return model(input_batch)
    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)
    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return sum([loss_fn[idx](output_batch[idx], target_batch[idx]) 
                    for idx in range(len(output_batch))])
    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)

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
    def grab_batch_from_loader(self, loader_iter):
        input_batch, target_batch = next(loader_iter)
        return [Variable(input_) for input_ in input_batch], Variable(target_batch)
    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        target_batch = tforms[1](target_batch)
        return input_batch, target_batch
    def forward_pass(self, input_batch, model):
        return model(*input_batch)
    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)
    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch, target_batch)
    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)

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
    def grab_batch_from_loader(self, loader_iter):
        input_batch, target_batch = next(loader_iter)
        return [Variable(input_) for input_ in input_batch], [Variable(target_) for target_ in target_batch]
    def apply_transforms(self, tforms, input_batch, target_batch):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        target_batch = [tforms[1](target_) for target_ in target_batch]
        return input_batch, target_batch
    def forward_pass(self, input_batch, model):
        return model(*input_batch)
    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)
    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return sum([loss_fn[idx](output_batch[idx], target_batch[idx]) 
                    for idx in range(len(output_batch))])
    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)

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
    def grab_batch_from_loader(self, loader_iter):
        input_batch = next(loader_iter)
        return Variable(input_batch), None
    def apply_transforms(self, tforms, input_batch, target_batch=None):
        input_batch = tforms[0](input_batch)
        return input_batch, None
    def forward_pass(self, input_batch, model):
        return model(input_batch)
    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)
    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch)
    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)

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
    def grab_batch_from_loader(self, loader_iter):
        input_batch = next(loader_iter)
        return [Variable(input_) for input_ in input_batch], None
    def apply_transforms(self, tforms, input_batch, target_batch=None):
        input_batch = [tforms[0](input_) for input_ in input_batch]
        return input_batch, None
    def forward_pass(self, input_batch, model):
        return model(*input_batch)
    def get_partial_forward_fn(self, model):
        return functools.partial(self.forward_pass, model=model)
    def calculate_loss(self, output_batch, target_batch, loss_fn):
        return loss_fn(output_batch)
    def get_partial_loss_fn(self, loss_fn):
        return functools.partial(self.calculate_loss, loss_fn=loss_fn)
