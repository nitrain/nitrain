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
                     _get_current_time, _nb_function_args,
                     _standardize_user_data)

from ..callbacks import CallbackModule, History, TQDM
from ..regularizers import RegularizerContainer
from ..initializers import InitializerContainer
from ..constraints import ConstraintContainer


class ModuleTrainer(object):

    def __init__(self, model):
        """
        ModelTrainer for high-level training of Pytorch models
        """
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

    def compile(self,
                optimizer,
                loss,
                regularizers=None,
                initializers=None,
                constraints=None,
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
        #print('TRAINING MODEL\n\n')

        inputs, targets = _standardize_user_data(inputs, targets)
        has_target = targets is not None
        if has_target:
            nb_targets = len(targets)

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
                'has_validation_data': False
            }
            callbacks.on_train_begin(logs=train_begin_logs)

            # calculate total number of batches
            nb_batches = int(math.ceil(inputs[0].size(0) / batch_size))

            # loop through each epoch
            for epoch_idx in range(nb_epoch):
                epoch_logs = {
                    'nb_batches': nb_batches,
                    'nb_epoch': nb_epoch,
                    'has_validation_data': False
                }
                callbacks.on_epoch_begin(epoch_idx, epoch_logs)

                # shuffle inputs and targets if necessary
                if shuffle:
                    rand_idx = th.randperm(len(inputs[0]))
                    inputs = [ins[rand_idx] for ins in inputs]
                    targets = [tars[rand_idx] for tars in targets]

                # loop through each batch
                for batch_idx in range(nb_batches):
                    self._REGULARIZER_CONTAINER.reset()

                    batch_logs = {'batch_idx': batch_idx}
                    callbacks.on_batch_begin(batch_idx, batch_logs) 

                    # grab an input batch and a target batch if necessary
                    input_batch = [Variable(x[batch_idx*batch_size:(batch_idx+1)*batch_size]) for x in inputs]
                    if has_target:
                        target_batch = [Variable(y[batch_idx*batch_size:(batch_idx+1)*batch_size]) for y in targets]

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
                        regularizer_loss = self._REGULARIZER_CONTAINER.get_value()
                        loss += regularizer_loss
                        batch_logs['regularizer_loss'] = regularizer_loss.data[0]

                    batch_logs['loss'] = loss.data[0]

                    # backward pass and optimizer step
                    loss.backward()
                    self._optimizer.step()

                    callbacks.on_batch_end(batch_idx, batch_logs)
                    # apply batch constraints
                    if self._has_constraints:
                        self._CONSTRAINT_CONTAINER.apply_constraints()
                
                # END OF EPOCH
                epoch_logs.update(self.history.batch_metrics)
                callbacks.on_epoch_end(epoch_idx, epoch_logs)

                # exit the training loop if necessary (e.g. EarlyStopping)
                if self._stop_training:
                    break

        train_logs = {
            'final_loss': self.history.losses[-1],
            'best_loss': min(self.history.losses),
            'end_time': _get_current_time()
        }

        callbacks.on_train_end(logs=train_logs)

    def save_state_dict(self, file):
        """
        Save a model parameters to disk
        """
        # model parameters -> ordered dict
        state_dict = self.model.state_dict()
        th.save(state_dict, file)







