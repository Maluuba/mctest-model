from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.callbacks import Callback
import logging
from save_model import save_weights

logger = logging.getLogger(__name__)


class LearningRateCutting(Callback):
    '''Stop training when a monitored quantity has stopped improving.
    '''

    def __init__(self, monitor='val_loss', cut_ratio=0.5, patience=2, scheduled_start_epoch=1, scheduled_cut_ratio=1.):
        """
        Args:
            monitor: quantity to be monitored.
            cut_ratio: cut the learning rate by this percent.
            patience: number of epochs with no improvement
                after which training will be stopped.
            scheduled_start_epoch: from which epoch to do scheduled learning rate discount
            scheduled_cut_ratio: learning rate discount ratio.
        """
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.best = np.Inf
        self.wait = 0
        self.cut_ratio = cut_ratio
        self.monitor_decrease = False
        self.scheduled_start_epoch = scheduled_start_epoch
        self.scheduled_cut_ratio = scheduled_cut_ratio

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            logger.warn('Cutting learning rate requires %s available!' % self.monitor)

        # schedule decay test.
        if epoch >= self.scheduled_start_epoch:
            current_lr = self.model.optimizer.lr.get_value()
            self.model.optimizer.lr.set_value(np.float32(current_lr * self.scheduled_cut_ratio))
            logger.info('learning decay (%s) by schedule at epoch %s' % (self.scheduled_cut_ratio, epoch))

        if current < self.best:
            self.best = current
            self.wait = 0
            self.monitor_decrease = True
        else:
            self.monitor_decrease = False
            if self.wait >= self.patience:
                logger.info('cutting learning rate by %s' % self.cut_ratio)
                current_lr = self.model.optimizer.lr.get_value()
                self.model.optimizer.lr.set_value(current_lr * self.cut_ratio)
                self.wait = 0
            else:
                self.wait += 1


class Evaluation(Callback):
    '''eval on test set, particularly for using generator.
    '''
    def __init__(self, test_data, mode='max', monitor='acc', name='test'):
        super(Evaluation, self).__init__()

        self.test_data = test_data
        self.monitor = monitor
        self.wait = 0
        self.name = name

        if mode not in ['auto', 'min', 'max']:
            logger.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode), RuntimeWarning)
            mode = 'auto'

        self.test_result = -1
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        # fetch the val_ measures.
        measures = [key for key in logs.keys() if self.monitor in key and 'val_' in key]
        assert len(measures) == 1
        if not measures:
            logger.warn('requires %s available!' % self.monitor, RuntimeWarning)

        measure = measures[0]
        score = logs[measure]
        x, y = self.test_data
        results = self.model.evaluate(x, y, verbose=0)
        logger.info("\n=== Results on %s ===\n %s " % (self.name, zip(self.model.metrics_names, results)))
        if self.monitor_op(score, self.best):
            self.best = score
            for metric, result in zip(self.model.metrics_names, results):
                if self.monitor in metric:
                    self.test_result = result

    def on_train_end(self, logs={}):
        logger.info('best valid/test are %s, %s' % (self.best, self.test_result))


class ModelCheckpoint(Callback):
    '''Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then multiple files will be save with the epoch number and
    the validation loss.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the validation loss will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minization of the monitored. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.

    '''
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto'):

        super(Callback, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    save_weights(self.model, filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            save_weights(self.model, filepath, overwrite=True)
