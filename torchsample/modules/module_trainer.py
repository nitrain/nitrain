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
        self.callbacks = [self.history]

        # regularizers
        self.regularizers = []
        self._has_regularizers = False

        # initializers
        self.initializers = []
        self._has_initializers = False

        # constraints
        self.constraints = []
        self._has_constraints = False

        # metrics
        self.metrics = []
        self._has_metrics = False

        # transforms
        self.transforms = []
        self._has_transforms = False

        # losses
        self.loss = None
        self._loss_fn = None

        # other properties
        self._stop_training = False

    def set_loss(self, loss):
        self.loss = loss
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

    def fit(self,
            inputs,
            targets=None,
            nb_epoch=100,
            batch_size=32,
            shuffle=False,
            cuda_device=-1,
            verbose=1):
        num_inputs, num_targets = _parse_num_inputs_and_targets(inputs, targets)
        fitter = _get_fitter(self, num_inputs, num_targets)

        if cuda_device > -1:
            inputs, targets = fitter.move_to_cuda(cuda_device, inputs, targets)

        len_inputs = len(inputs) if not _is_iterable(inputs) else len(inputs[0])
        nb_batches = int(math.ceil(len_inputs / batch_size))

        for epoch_idx in range(nb_epoch):
            if shuffle:
                inputs, targets = fitter.shuffle_arrays(inputs, targets)

            for batch_idx in range(nb_batches):
                input_batch, target_batch = fitter.grab_batch(batch_idx, batch_size, inputs, targets)

                self._optimizer.zero_grad()
                output_batch = fitter.forward_pass(self.model, input_batch)
                loss = fitter.calculate_loss(self._loss_fn, output_batch, target_batch)
                loss.backward()
                self._optimizer.step()


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
        predictor = _get_fitter(self, num_inputs, num_targets=0)

        if cuda_device >= 0:
            inputs = predictor.move_to_cuda(cuda_device, inputs)
            self.model.cuda(cuda_device)

        nb_batches = int(math.ceil(len_inputs / batch_size))
        
        # get first batch
        input_batch, _ = predictor.grab_batch(0, batch_size, inputs)
        output_batch = predictor.forward_pass(self.model, input_batch)

        for batch_idx in range(nb_batches):
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
        evaluator = _get_fitter(self, num_inputs, num_targets)

        if cuda_device > -1:
            inputs, targets = evaluator.move_to_cuda(cuda_device, inputs, targets)

        len_inputs = len(inputs) if not _is_iterable(inputs) else len(inputs[0])
        nb_batches = int(math.ceil(len_inputs / batch_size))

        total_losses = []
        for batch_idx in range(nb_batches):
            input_batch, target_batch = evaluator.grab_batch(batch_idx, batch_size, inputs, targets)

            self._optimizer.zero_grad()
            output_batch = evaluator.forward_pass(self.model, input_batch)
            loss = evaluator.calculate_loss(self._loss_fn, output_batch, target_batch)
            total_losses.append(loss.data[0])
        total_loss = th.mean(th.FloatTensor(total_losses))
        return total_loss


def _get_fitter(trainer, num_inputs, num_targets):
    if (num_inputs == 1) and (num_targets == 1):
        fitter = SingleInput_SingleTarget_Fitter()

    elif (num_inputs == 1) and (num_targets > 1):
        # use same loss function for all targets if multiple loss fns not explicitly given
        if not _is_iterable(trainer._loss_fn):
            trainer._loss_fn = [trainer._loss_fn] * num_targets
        else:
            if len(trainer._loss_fn) != num_targets:
                raise ValueError('must give one loss function for every input if you give multiple')
        fitter = SingleInput_MultiTarget_Fitter()

    elif (num_inputs == 1) and (num_targets == 0):
        fitter = SingleInput_NoTarget_Fitter()

    elif (num_inputs > 1) and (num_targets == 1):
        fitter = MultiInput_SingleTarget_Fitter()

    elif (num_inputs > 1) and (num_targets > 1):
        # use same loss function for all targets if multiple loss fns not explicitly given
        if not _is_iterable(trainer._loss_fn):
            trainer._loss_fn = [trainer._loss_fn] * num_targets
        else:
            if len(trainer._loss_fn) != num_targets:
                raise ValueError('must give one loss function for every input if you give multiple')
        fitter = MultiInput_MultiTarget_Fitter()

    elif (num_inputs > 1) and (num_targets == 0):
        fitter = MultiInput_NoTarget_Fitter()

    return fitter


class SingleInput_SingleTarget_Fitter(object):
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


class SingleInput_MultiTarget_Fitter(object):
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


class MultiInput_SingleTarget_Fitter(object):
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


class MultiInput_MultiTarget_Fitter(object):
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


class SingleInput_NoTarget_Fitter(object):
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


class MultiInput_NoTarget_Fitter(object):
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
