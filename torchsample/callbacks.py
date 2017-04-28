"""
SuperModule Callbacks
"""

from __future__ import absolute_import
from __future__ import print_function

from collections import OrderedDict
from collections import Iterable
import warnings

import os
import csv
import time
from tqdm import tqdm

import torch


class CallbackModule(object):
    """
    Container holding a list of callbacks.
    """

    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class TQDM(Callback):

    def __init__(self):
        """
        TQDM Progress Bar callback

        This callback is automatically applied to 
        every SuperModule if verbose > 0
        """
        super(TQDM, self).__init__()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if self.progbar:
            self.progbar.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.progbar = tqdm(total=logs['nb_batches'],
                            unit=' batches')
        self.progbar.set_description('Epoch %i/%i' % 
                        (epoch+1, logs['nb_epoch']))

    def on_epoch_end(self, epoch, logs=None):
        log_data = {key: '%.04f' % value for key, value in self.model.history.batch_metrics.items()}
        for k, v in logs.items():
            if k.endswith('metric'):
                log_data[k.split('_metric')[0]] = '%.02f' % v
        self.progbar.set_postfix(log_data)
        self.progbar.update()
        self.progbar.close()

    def on_batch_begin(self, batch, logs=None):
        self.progbar.update(1)

    def on_batch_end(self, batch, logs=None):
        log_data = {key: '%.04f' % value for key, value in self.model.history.batch_metrics.items()}
        for k, v in logs.items():
            if k.endswith('metric'):
                log_data[k.split('_metric')[0]] = '%.02f' % v
        self.progbar.set_postfix(log_data)


class History(Callback):
    """
    Callback that records events into a `History` object.

    This callback is automatically applied to
    every SuperModule.
    """
    def __init__(self):
        super(History, self).__init__()
        self.seen = 0.

    def on_train_begin(self, logs=None):
        self.epoch_metrics = {
            'epochs': [],
            'losses': [],
            'regularizer_losses': [],
            'constraint_losses': [],
            'val_losses': []
        }
        #if logs['has_validation_data']:
        #    self.batch_metrics['val_loss'] = 0.

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_metrics = {
            'loss': 0.
        }
        if self.model._has_regularizers:
            self.batch_metrics['regularizer_loss'] = 0.
        if self.model._has_constraints:
            self.batch_metrics['constraint_loss'] = 0.
        self.seen = 0.

    def on_batch_end(self, batch, logs=None):
        for k in self.batch_metrics:
            self.batch_metrics[k] = (self.seen*self.batch_metrics[k] + logs[k]*logs['batch_samples']) / (self.seen+logs['batch_samples'])
        self.seen += logs['batch_samples']

class ModelCheckpoint(Callback):
    """
    Model Checkpoint to save model weights during training
    """

    def __init__(self, 
                 file, 
                 monitor='val_loss', 
                 save_best_only=False, 
                 save_weights_only=True,
                 max_checkpoints=-1,
                 verbose=0):
        """
        Model Checkpoint to save model weights during training

        Arguments
        ---------
        file : string
            file to which model will be saved.
            It can be written 'filename_{epoch}_{loss}' and those
            values will be filled in before saving.
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        save_best_only : boolean
            whether to only save if monitored value has improved
        save_weight_only : boolean 
            whether to save entire model or just weights
            NOTE: only `True` is supported at the moment
        max_checkpoints : integer > 0 or -1
            the max number of models to save. Older model checkpoints
            will be overwritten if necessary. Set equal to -1 to have
            no limit
        verbose : integer in {0, 1}
            verbosity
        """
        self.file = file
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.max_checkpoints = max_checkpoints
        self.verbose = verbose

        if self.max_checkpoints > 0:
            self.old_files = []

        # mode = 'min' only supported
        self.best_loss = 1e15
        super(ModelCheckpoint, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        file = self.file.format(epoch='%03i'%(epoch+1), 
                                loss='%0.4f'%logs[self.monitor])
        if self.save_best_only:
            current_loss = logs.get(self.monitor)
            if current_loss is None:
                pass
            else:
                if current_loss < self.best_loss:
                    if self.verbose > 0:
                        print('\nEpoch %i: improved from %0.4f to %0.4f saving model to %s' % 
                              (epoch+1, self.best_loss, current_loss, file))
                    self.best_loss = current_loss
                    self.model.save_state_dict(file)
                    if self.max_checkpoints > 0:
                        if len(self.old_files) == self.max_checkpoints:
                            try:
                                os.remove(self.old_files[0])
                            except:
                                pass
                            self.old_files = self.old_files[1:]
                        self.old_files.append(file)
        else:
            if self.verbose > 0:
                print('\nEpoch %i: saving model to %s' % (epoch+1, file))
            self.model.save_state_dict(file)
            if self.max_checkpoints > 0:
                if len(self.old_files) == self.max_checkpoints:
                    try:
                        os.remove(self.old_files[0])
                    except:
                        pass
                    self.old_files = self.old_files[1:]
                self.old_files.append(file)


class EarlyStopping(Callback):
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self, 
                 monitor='val_loss',
                 min_delta=0,
                 patience=0):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs

        Arguments
        ---------
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvment before terminating.
            the counter be reset after each improvment
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e-15
        self.stopped_epoch = 0
        super(EarlyStopping, self).__init__()

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_loss = 1e15

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    self.model._stop_training = True
                self.wait += 1

    def on_train_end(self, logs):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping at Epoch %04i' % 
                (self.stopped_epoch))


class LearningRateScheduler(Callback):
    """
    Schedule the learning rate according to some function of the 
    current epoch index, current learning rate, and current train/val loss.
    """

    def __init__(self, schedule):
        """
        LearningRateScheduler callback to adapt the learning rate
        according to some function

        Arguments
        ---------
        schedule : callable
            should return a number of learning rates equal to the number
            of optimizer.param_groups. It should take the epoch index and
            **kwargs (or logs) as argument. **kwargs (or logs) will return
            the epoch logs such as mean training and validation loss from
            the epoch
        """
        self.schedule = schedule
        super(LearningRateScheduler, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        current_lrs = [p['lr'] for p in self.model._optimizer.param_groups]
        lr_list = self.schedule(epoch, current_lrs, **logs)
        if not isinstance(lr_list, list):
            lr_list = [lr_list]

        for param_group, lr_change in zip(self.model._optimizer.param_groups, lr_list):
            param_group['lr'] = lr_change


class ReduceLROnPlateau(Callback):
    """
    Reduce the learning rate if the train or validation loss plateaus
    """

    def __init__(self,
                 monitor='val_loss', 
                 factor=0.1, 
                 patience=10,
                 epsilon=0, 
                 cooldown=0, 
                 min_lr=0,
                 verbose=0):
        """
        Reduce the learning rate if the train or validation loss plateaus

        Arguments
        ---------
        monitor : string in {'loss', 'val_loss'}
            which metric to monitor
        factor : floar
            factor to decrease learning rate by
        patience : integer
            number of epochs to wait for loss improvement before reducing lr
        epsilon : float
            how much improvement must be made to reset patience
        cooldown : integer 
            number of epochs to cooldown after a lr reduction
        min_lr : float
            minimum value to ever let the learning rate decrease to
        verbose : integer
            whether to print reduction to console
        """
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.wait = 0
        self.best_loss = 1e15
        self._reset()
        super(ReduceLROnPlateau, self).__init__()

    def _reset(self):
        """
        Reset the wait and cooldown counters
        """
        self.monitor_op = lambda a, b: (a - b) < -self.epsilon
        self.best_loss = 1e15
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = [p['lr'] for p in self.model._optimizer.param_groups]
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            pass
        else:  
            if self.cooldown_counter > 0:
                # if in cooldown phase
                self.cooldown_counter += 1
                self.wait = 0
            if self.monitor_op(current_loss, self.best_loss):
                # not in cooldown and loss improved
                self.best_loss = current_loss
                self.wait = 0
            else:
                # loss didnt improve
                for p in self.model._optimizer.param_groups:
                    old_lr = p['lr']
                    if old_lr > self.min_lr + 1e-4:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: reducing lr from %0.3f to %0.3f' % 
                                (epoch, old_lr, new_lr))
                        p['lr'] = new_lr
                        self.cooldown_counter = self.cooldown
                        self.wait = 0


class CSVLogger(Callback):
    """
    Logs epoch-level metrics to a CSV file
    """

    def __init__(self, 
                 file, 
                 separator=',', 
                 append=False):
        """
        Logs epoch-level metrics to a CSV file

        Arguments
        ---------
        file : string
            path to csv file
        separator : string
            delimiter for file
        apped : boolean
            whether to append result to existing file or make new file
        """
        self.file = file
        self.sep = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.file):
                with open(self.file) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.file, 'a')
        else:
            self.csv_file = open(self.file, 'w')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        RK = {'nb_batches', 'nb_epoch'}

        def handle_value(k):
            is_zero_dim_tensor = isinstance(k, torch.Tensor) and k.dim() == 0
            if isinstance(k, Iterable) and not is_zero_dim_tensor:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                    fieldnames=['epoch'] + [k for k in self.keys if k not in RK], 
                    dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys if key not in RK)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


class LambdaCallback(Callback):
    """
    Callback for creating simple, custom callbacks on-the-fly.
    """
    def __init__(self,
                 on_epoch_begin=None,
                 on_epoch_end=None,
                 on_batch_begin=None,
                 on_batch_end=None,
                 on_train_begin=None,
                 on_train_end=None,
                 **kwargs):
        super(LambdaCallback, self).__init__()
        self.__dict__.update(kwargs)
        if on_epoch_begin is not None:
            self.on_epoch_begin = on_epoch_begin
        else:
            self.on_epoch_begin = lambda epoch, logs: None
        if on_epoch_end is not None:
            self.on_epoch_end = on_epoch_end
        else:
            self.on_epoch_end = lambda epoch, logs: None
        if on_batch_begin is not None:
            self.on_batch_begin = on_batch_begin
        else:
            self.on_batch_begin = lambda batch, logs: None
        if on_batch_end is not None:
            self.on_batch_end = on_batch_end
        else:
            self.on_batch_end = lambda batch, logs: None
        if on_train_begin is not None:
            self.on_train_begin = on_train_begin
        else:
            self.on_train_begin = lambda logs: None
        if on_train_end is not None:
            self.on_train_end = on_train_end
        else:
            self.on_train_end = lambda logs: None

