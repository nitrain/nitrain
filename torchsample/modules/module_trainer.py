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

# local imports
from ._utils import (_validate_loss_input, _validate_metric_input, 
                     _validate_optimizer_input, _validate_initializer_input,
                     _get_current_time, _standardize_user_data)

from ..callbacks import CallbackContainer, History, TQDM
from ..regularizers import RegularizerContainer
from ..initializers import InitializerContainer
from ..constraints import ConstraintContainer
from ..metrics import MetricsContainer


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

    def set_initializers(self, initializers):
        if not isinstance(initializers, (list,tuple)):
            initializers = [initializers]
        initializers = [_validate_initializer_input(it) for it in initializers]
        self._has_initializers = True
        self._initializers = initializers

    def set_constraints(self, constraints):
        if not isinstance(constraints, (list,tuple)):
            constraints = [constraints]
        self._has_constraints = True
        self._constraints = constraints

    def set_metrics(self, metrics):
        if not isinstance(metrics, (list,tuple)):
            metrics = [metrics]
        metrics = [_validate_metric_input(m) for m in metrics]
        self._has_metrics = True
        self._metrics = metrics

    def set_callbacks(self, callbacks):
        self._callbacks += callbacks

    def add_callback(self, callback):
        self._callbacks.append(callback)

    def set_transforms(self, transforms):
        if not isinstance(transforms, (list,tuple)):
            transforms = [transforms, None, None]
        if len(transforms) == 1:
            transforms = [transforms, None, None]
        elif len(transforms) == 2:
            transforms = [transforms, transforms, None]

        self._has_input_transform = transforms[0] is not None
        self._has_target_transform = transforms[1] is not None
        self._has_co_transform = transforms[2] is not None

        self._has_transforms = True
        self._transforms = transforms

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

        if regularizers is not None:
            self.set_regularizers(regularizers)
            self._REGULARIZER_CONTAINER = RegularizerContainer(self._regularizers)
            self._REGULARIZER_CONTAINER.register_forward_hooks(self.model)

        if initializers is not None:
            self.set_initializers(initializers)
            self._INITIALIZER_CONTAINER = InitializerContainer(self._initializers)
            self._INITIALIZER_CONTAINER(self.model)

        if constraints is not None:
            self.set_constraints(constraints)
            self._CONSTRAINT_CONTAINER = ConstraintContainer(self._constraints)
            self._CONSTRAINT_CONTAINER.register_constraints(self.model)
        
        if metrics is not None:
            self.set_metrics(metrics)
            self._METRIC_CONTAINER = MetricsContainer(self._metrics)

        if callbacks is not None:
            self.set_callbacks(callbacks)

        if transforms is not None:
            self.set_transforms(transforms)

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

        # enter context-manager for progress bar
        with TQDM() as pbar:
            # create callbacks
            progressbar = []
            # add progress bar if necessary
            if verbose > 0:
                progressbar = [pbar]
            _CALLBACK_CONTAINER = CallbackContainer(self._callbacks + progressbar)
            _CALLBACK_CONTAINER.set_model(self)

            train_logs = {
                'start_time': _get_current_time(),
                'has_validation_data': False
            }
            _CALLBACK_CONTAINER.on_train_begin(logs=train_logs)

            # calculate total number of batches
            nb_batches = int(math.ceil(inputs[0].size(0) / batch_size))

            # loop through each epoch
            for epoch_idx in range(nb_epoch):
                epoch_logs = {
                    'nb_batches': nb_batches,
                    'nb_epoch': nb_epoch,
                    'has_validation_data': False
                }
                _CALLBACK_CONTAINER.on_epoch_begin(epoch_idx, epoch_logs)

                # reset metric counts
                if self._has_metrics:
                    self._METRIC_CONTAINER.reset()

                # shuffle inputs and targets if necessary
                if shuffle:
                    rand_indices = th.randperm(len(inputs[0]))
                    inputs = [ins[rand_indices] for ins in inputs]
                    targets = [tars[rand_indices] for tars in targets]

                # loop through each batch
                for batch_idx in range(nb_batches):
                    # reset regularizer each batch
                    self._REGULARIZER_CONTAINER.reset()

                    batch_logs = {'batch_idx': batch_idx}
                    _CALLBACK_CONTAINER.on_batch_begin(batch_idx, batch_logs) 

                    # grab an input batch and a target batch if necessary
                    input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size]) for x in inputs]
                    target_batch = [Variable(y[batch_idx*batch_size:(batch_idx+1)*batch_size]) for y in targets]

                    # run transforms if necessary
                    if self._has_input_transform:
                        input_batch = [self._transforms[0](x) for x in inputs]
                    if self._has_target_transform:
                        target_batch = [self._transforms[1](y) for y in targets]
                    if self._has_co_transform:
                        input_batch, target_batch = zip(*[self._transforms[2](x, y) 
                            for x, y in zip(input_batch, target_batch)])

                    batch_logs['batch_samples'] = len(input_batch[0])

                    # zero grads and forward pass
                    self._optimizer.zero_grad()
                    output_batch = self.model(*input_batch)

                    # apply multiple loss functions if necessary
                    if not isinstance(output_batch, (list,tuple)):
                        output_batch = [output_batch]

                    # multiple outputs, but they all go into one loss function
                    loss = sum([self._loss_fns[loss_idx](output_batch[loss_idx], target_batch[loss_idx]) 
                                    for loss_idx in range(nb_targets)])
                    # add regularizers to loss if necessary
                    if self._has_regularizers:
                        regularizer_loss = self._REGULARIZER_CONTAINER.get_value()
                        loss += regularizer_loss
                        batch_logs['regularizer_loss'] = regularizer_loss.data[0]

                    # calculate metrics if necessary
                    if self._has_metrics:
                        metrics_logs = self._METRIC_CONTAINER(output_batch[0], target_batch[0])
                        batch_logs.update(metrics_logs)

                    batch_logs['loss'] = loss.data[0]

                    # backward pass and optimizer step
                    loss.backward()
                    self._optimizer.step()

                    # apply batch constraints
                    if self._has_constraints:
                        self._CONSTRAINT_CONTAINER.apply_batch_constraints(batch_idx)

                    _CALLBACK_CONTAINER.on_batch_end(batch_idx, batch_logs)

                # END OF EPOCH
                if self._has_constraints:
                    self._CONSTRAINT_CONTAINER.apply_epoch_constraints(epoch_idx)

                epoch_logs.update(self.history.batch_metrics)
                _CALLBACK_CONTAINER.on_epoch_end(epoch_idx, epoch_logs)

                # exit the training loop if necessary (e.g. EarlyStopping)
                if self._stop_training:
                    break

        train_logs['final_loss'] = self.history.losses[-1],
        train_logs['best_loss'] = min(self.history.losses),
        train_logs['stop_time'] = _get_current_time()

        _CALLBACK_CONTAINER.on_train_end(logs=train_logs)

    def save_state_dict(self, file):
        """
        Save a model parameters to disk
        """
        # model parameters -> ordered dict
        state_dict = self.model.state_dict()
        th.save(state_dict, file)







