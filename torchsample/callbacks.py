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
                                         (epoch + 1, self.train_logs['num_epoch']))
        except:
            pass

    def on_epoch_end(self, epoch, logs=None):
        log_data = {}

        for k, v in logs.items():
            if k.endswith('metric'):
                log_data[k.split('_metric')[0]] = '%.02f' % v
            else:
                log_data[k] = '%.04f' % v
        # print(log_data)
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
        self.epoch_metrics.update(self.batch_metrics)
        # TODO
        pass

    def on_batch_end(self, batch, logs=None):

        for k in self.batch_metrics:
            self.batch_metrics[k] = (self.samples_seen * self.batch_metrics[k] + logs[k] * self.batch_size) / (
                    self.samples_seen + self.batch_size)

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
            # 'arch': args.arch,
            'state_dict': self.trainer.model.state_dict(),
            # 'best_prec1': best_prec1,
            'optimizer': self.trainer._optimizer.state_dict(),
            # 'loss':{},
            #            #'regularizers':{},
            #            #'constraints':{},
            #            #'initializers':{},
            #            #'metrics':{},
            #            #'val_loss':{}
        }, file)
        if is_best:
            shutil.copyfile(file, 'model_best.pth.tar')

    def on_epoch_end(self, epoch, logs=None):

        file = self.file.format(epoch='%03i' % (epoch + 1),
                                loss='%0.4f' % logs[self.monitor])
        if self.save_best_only:
            current_loss = logs.get(self.monitor)
            if current_loss is None:
                pass
            else:
                if current_loss < self.best_loss:
                    if self.verbose > 0:
                        print('\nEpoch %i: improved from %0.4f to %0.4f saving model to %s' %
                              (epoch + 1, self.best_loss, current_loss, file))
                    self.best_loss = current_loss
                    # if self.save_weights_only:
                    # else:
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
                print('\nEpoch %i: saving model to %s' % (epoch + 1, file))
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
            # print("\nCurrent loss improvement {}".format(current_loss - self.best_loss))
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            # print("\nCurrent Wait {}".format(self.wait))

            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    print("\nTerminated Training for Early Stopping at Epoch {}".format(
                        (self.stopped_epoch)))
                    self.trainer._stop_training = True

                self.wait += 1
                # print("\nCurrent Wait {}".format(self.wait))

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
            print("burayra girdim")
            schedule = self.schedule_from_dict
            self.schedule_dict = schedule
            if any([k < 1.0 for k in schedule.keys()]):
                self.fractional_bounds = False
            else:
                self.fractional_bounds = True
        self.schedule = schedule
        self.best_val_loss = 1e-15
        super(LRScheduler, self).__init__()

    def schedule_from_dict(self, epoch, logs=None):
        for epoch_bound, learn_rate in self.schedule_dict.items():
            # epoch_bound is in units of "epochs"
            if not self.fractional_bounds:
                if epoch_bound < epoch:
                    return learn_rate
            # epoch_bound is in units of "cumulative percent of epochs"
            else:
                if epoch <= epoch_bound * logs['num_epoch']:
                    return learn_rate
        warnings.warn('Check the keys in the schedule dict.. Returning last value')
        return learn_rate

    def on_train_begin(self, logs=None):
        self.best_val_loss = 1e15

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_lrs = [p['lr'] for p in self.trainer._optimizer.param_groups]
        current_val_loss = logs.get("val_loss")

        if (current_val_loss - self.best_val_loss) < 0:
            self.best_val_loss = current_val_loss

        lr_list = self.schedule(epoch, current_lrs, self.best_val_loss, current_val_loss)
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
        log_data = {}
        for k in logs:
            k_log = k.split('_metric')[0]
            log_data[k_log] = logs[k]
        logs = log_data
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
        REJECT_KEYS = {'has_validation_data'}
        MODEL_NAME = self.save_prefix + str(model_idx)  # figure out how to get model name
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
        REJECT_KEYS = {'has_validation_data'}
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


class CyclicLR(Callback):
    """
    Take is taken from: https://github.com/bckenstler/CLR/blob/master/clr_callback.py
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, schedule, base_lr=0.0001, max_lr=0.01, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()
        self.schedule = schedule
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        if self.clr_iterations == 0:
            initial_lr = self.base_lr
            if not isinstance(initial_lr, list):
                initial_lr = [initial_lr]
            for param_group, lr_change in zip(self.trainer._optimizer.param_groups, initial_lr):
                param_group['lr'] = lr_change
        else:
            new_lr = self.clr()
            if not isinstance(new_lr, list):
                new_lr = [new_lr]
            for param_group, lr_change in zip(self.trainer._optimizer.param_groups, new_lr):
                param_group['lr'] = lr_change

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        current_lr = [p['lr'] for p in self.trainer._optimizer.param_groups]

        self.history.setdefault('lr', []).append(current_lr[0])
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        new_lr = self.clr()
        if not isinstance(new_lr, list):
            new_lr = [new_lr]
        for param_group, lr_change in zip(self.trainer._optimizer.param_groups, new_lr):
            param_group['lr'] = lr_change

    def on_epoch_end(self, epoch, logs=None):
        # logs = logs or {}
        # self.schedule(epoch, [p['lr'] for p in self.trainer._optimizer.param_groups][0])
        pass

    def on_train_end(self, logs=None):
        print(self.history)
