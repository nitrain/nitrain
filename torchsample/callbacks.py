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
from tempfile import NamedTemporaryFile
import shutil
import datetime
import numpy as np

from tqdm import tqdm

import torch as th


def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")

class CallbackContainer(object):
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

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

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
        logs['start_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        logs['final_loss'] = self.trainer.history.epoch_losses[-1],
        logs['best_loss'] = min(self.trainer.history.epoch_losses),
        logs['stop_time'] = _get_current_time()
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

    def set_trainer(self, model):
        self.trainer = model

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
        self.progbar = None
        super(TQDM, self).__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the dbconnection gets closed
        if self.progbar is not None:
            self.progbar.close()

    def on_train_begin(self, logs):
        self.train_logs = logs

    def on_epoch_begin(self, epoch, logs=None):
        try:
            self.progbar = tqdm(total=self.train_logs['num_batches'],
                                unit=' batches')
            self.progbar.set_description('Epoch %i/%i' % 
                            (epoch+1, self.train_logs['num_epoch']))
        except:
            pass

    def on_epoch_end(self, epoch, logs=None):
        log_data = {key: '%.04f' % value for key, value in self.trainer.history.batch_metrics.items()}
        for k, v in logs.items():
            if k.endswith('metric'):
                log_data[k.split('_metric')[0]] = '%.02f' % v
            else:
                 log_data[k] = v
        self.progbar.set_postfix(log_data)
        self.progbar.update()
        self.progbar.close()

    def on_batch_begin(self, batch, logs=None):
        self.progbar.update(1)

    def on_batch_end(self, batch, logs=None):
        log_data = {key: '%.04f' % value for key, value in self.trainer.history.batch_metrics.items()}
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
    def __init__(self, model):
        super(History, self).__init__()
        self.samples_seen = 0.
        self.trainer = model

    def on_train_begin(self, logs=None):
        self.epoch_metrics = {
            'loss': []
        }
        self.batch_size = logs['batch_size']
        self.has_val_data = logs['has_val_data']
        self.has_regularizers = logs['has_regularizers']
        if self.has_val_data:
            self.epoch_metrics['val_loss'] = []
        if self.has_regularizers:
            self.epoch_metrics['reg_loss'] = []

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_metrics = {
            'loss': 0.
        }
        if self.has_regularizers:
            self.batch_metrics['reg_loss'] = 0.
        self.samples_seen = 0.

    def on_epoch_end(self, epoch, logs=None):
        #for k in self.batch_metrics:
        #    k_log = k.split('_metric')[0]
        # self.epoch_metrics.update(self.batch_metrics)
        # TODO
        pass

    def on_batch_end(self, batch, logs=None):
        for k in self.batch_metrics:
            self.batch_metrics[k] = (self.samples_seen*self.batch_metrics[k] + logs[k]*self.batch_size) / (self.samples_seen+self.batch_size)
        self.samples_seen += self.batch_size

    def __getitem__(self, name):
        return self.epoch_metrics[name]

    def __repr__(self):
        return str(self.epoch_metrics)

    def __str__(self):
        return str(self.epoch_metrics)


class ModelCheckpoint(Callback):
    """
    Model Checkpoint to save model weights during training

    save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }
    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        th.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    """

    def __init__(self,
                 directory, 
                 filename='ckpt.pth.tar', 
                 monitor='val_loss', 
                 save_best_only=False, 
                 save_weights_only=True,
                 max_save=-1,
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
        max_save : integer > 0 or -1
            the max number of models to save. Older model checkpoints
            will be overwritten if necessary. Set equal to -1 to have
            no limit
        verbose : integer in {0, 1}
            verbosity
        """
        if directory.startswith('~'):
            directory = os.path.expanduser(directory)
        self.directory = directory
        self.filename = filename
        self.file = os.path.join(self.directory, self.filename)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.max_save = max_save
        self.verbose = verbose

        if self.max_save > 0:
            self.old_files = []

        # mode = 'min' only supported
        self.best_loss = float('inf')
        super(ModelCheckpoint, self).__init__()

    def save_checkpoint(self, epoch, file, is_best=False):
        th.save({ 
            'epoch': epoch + 1,
             #'arch': args.arch,
            'state_dict': self.trainer.model.state_dict(),
            #'best_prec1': best_prec1,
            'optimizer' : self.trainer._optimizer.state_dict(),
            #'loss':{},
                #            #'regularizers':{},
                #            #'constraints':{},
                #            #'initializers':{},
                #            #'metrics':{},
                #            #'val_loss':{}
            }, file)
        if is_best:
            shutil.copyfile(file, 'model_best.pth.tar')

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
                    #if self.save_weights_only:
                    #else:
                    self.save_checkpoint(epoch, file)
                    if self.max_save > 0:
                        if len(self.old_files) == self.max_save:
                            try:
                                os.remove(self.old_files[0])
                            except:
                                pass
                            self.old_files = self.old_files[1:]
                        self.old_files.append(file)
        else:
            if self.verbose > 0:
                print('\nEpoch %i: saving model to %s' % (epoch+1, file))
            self.save_checkpoint(epoch, file)
            if self.max_save > 0:
                if len(self.old_files) == self.max_save:
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
                 patience=5):
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
                    self.trainer._stop_training = True
                self.wait += 1

    def on_train_end(self, logs):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping at Epoch %04i' % 
                (self.stopped_epoch))


class LRScheduler(Callback):
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
        if isinstance(schedule, dict):
            schedule = self.schedule_from_dict
            self.schedule_dict = schedule
            if any([k < 1.0 for k in schedule.keys()]):
                self.fractional_bounds = False
            else:
                self.fractional_bounds = True
        self.schedule = schedule
        super(LRScheduler, self).__init__()

    def schedule_from_dict(self, epoch, logs=None):
        for epoch_bound, learn_rate in self.schedule_dict.items():
            # epoch_bound is in units of "epochs"
            if not self.fractional_bounds:
                if epoch_bound < epoch:
                    return learn_rate
            # epoch_bound is in units of "cumulative percent of epochs"
            else:
                if epoch <= epoch_bound*logs['num_epoch']:
                    return learn_rate
        warnings.warn('Check the keys in the schedule dict.. Returning last value')
        return learn_rate

    def on_epoch_begin(self, epoch, logs=None):
        current_lrs = [p['lr'] for p in self.trainer._optimizer.param_groups]
        lr_list = self.schedule(epoch, current_lrs, **logs)
        if not isinstance(lr_list, list):
            lr_list = [lr_list]

        for param_group, lr_change in zip(self.trainer._optimizer.param_groups, lr_list):
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
        logs['lr'] = [p['lr'] for p in self.trainer._optimizer.param_groups]
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            pass
        else:
            # if in cooldown phase
            if self.cooldown_counter > 0: 
                self.cooldown_counter -= 1
                self.wait = 0
            # if loss improved, grab new loss and reset wait counter
            if self.monitor_op(current_loss, self.best_loss):
                self.best_loss = current_loss
                self.wait = 0
            # loss didnt improve, and not in cooldown phase
            elif not (self.cooldown_counter > 0):
                if self.wait >= self.patience:
                    for p in self.trainer._optimizer.param_groups:
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
                self.wait += 1


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
        RK = {'num_batches', 'num_epoch'}

        def handle_value(k):
            is_zero_dim_tensor = isinstance(k, th.Tensor) and k.dim() == 0
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


class ExperimentLogger(Callback):

    def __init__(self,
                 directory,
                 filename='Experiment_Logger.csv',
                 save_prefix='Model_', 
                 separator=',', 
                 append=True):

        self.directory = directory
        self.filename = filename
        self.file = os.path.join(self.directory, self.filename)
        self.save_prefix = save_prefix
        self.sep = separator
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        super(ExperimentLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            open_type = 'a'
        else:
            open_type = 'w'

        # if append is True, find whether the file already has header
        num_lines = 0
        if self.append:
            if os.path.exists(self.file):
                with open(self.file) as f:
                    for num_lines, l in enumerate(f):
                        pass
                    # if header exists, DONT append header again
                with open(self.file) as f:
                    self.append_header = not bool(len(f.readline()))
                
        model_idx = num_lines
        REJECT_KEYS={'has_validation_data'}
        MODEL_NAME = self.save_prefix + str(model_idx) # figure out how to get model name
        self.row_dict = OrderedDict({'model': MODEL_NAME})
        self.keys = sorted(logs.keys())
        for k in self.keys:
            if k not in REJECT_KEYS:
                self.row_dict[k] = logs[k]

        class CustomDialect(csv.excel):
            delimiter = self.sep

        with open(self.file, open_type) as csv_file:
            writer = csv.DictWriter(csv_file,
                fieldnames=['model'] + [k for k in self.keys if k not in REJECT_KEYS], 
                dialect=CustomDialect)
            if self.append_header:
                writer.writeheader()

            writer.writerow(self.row_dict)
            csv_file.flush()

    def on_train_end(self, logs=None):
        REJECT_KEYS={'has_validation_data'}
        row_dict = self.row_dict

        class CustomDialect(csv.excel):
            delimiter = self.sep
        self.keys = self.keys
        temp_file = NamedTemporaryFile(delete=False, mode='w')
        with open(self.file, 'r') as csv_file, temp_file:
            reader = csv.DictReader(csv_file,
                fieldnames=['model'] + [k for k in self.keys if k not in REJECT_KEYS], 
                dialect=CustomDialect)
            writer = csv.DictWriter(temp_file,
                fieldnames=['model'] + [k for k in self.keys if k not in REJECT_KEYS], 
                dialect=CustomDialect)
            for row_idx, row in enumerate(reader):
                if row_idx == 0:
                    # re-write header with on_train_end's metrics
                    pass
                if row['model'] == self.row_dict['model']:
                    writer.writerow(row_dict)
                else:
                    writer.writerow(row)
        shutil.move(temp_file.name, self.file)   


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

