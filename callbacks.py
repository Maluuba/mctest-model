from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras.callbacks import Callback
import logging

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
    def __init__(self, test_data, mode='max', monitor='acc'):
        super(Evaluation, self).__init__()

        self.test_data = test_data
        self.monitor = monitor
        self.wait = 0

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
        logger.info("test results is: %s " % zip(self.model.metrics_names, results))
        if self.monitor_op(score, self.best):
            self.best = score
            for metric, result in zip(self.model.metrics_names, results):
                if self.monitor in metric:
                    self.test_result = result

    def on_train_end(self, logs={}):
        logger.info('best valid/test are %s, %s' % (self.best, self.test_result))
