# encoding: utf-8
try:
    import gpuselect
except:
    print('gpuselect is not installed')
import os
import logging
import numpy as np

from keras.callbacks import EarlyStopping
from setup_logger import setup_logging

from model import PHM
from callbacks import LearningRateCutting, Evaluation

logger = logging.getLogger(__name__)


class McDataset(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset_name = 'mctest'
        self._load()

    def _load(self):
        import h5py
        print 'init dataset with h5 file.'
        meta_data = {}
        f = h5py.File(self.data_path, 'r')
        dataset = f[self.dataset_name]
        for key in dataset.attrs:
            meta_data[key] = dataset.attrs[key]

        words_flatten = f['words_flatten'][0]
        id2word = words_flatten.split('\n')
        # assert len(self.id2word) == f.attrs['vocab_len'], "%s != %s" % (len(id2word), f.attrs['vocab_len'])
        word2id = dict(zip(id2word, range(len(id2word))))
        meta_data['id2word'] = id2word
        meta_data['word2id'] = word2id
        del words_flatten
        meta_data['idfs'] = dataset['idfs'][:]

        data = {}
        for key in dataset['data']:
            data[key] = {}
            for inner_key in dataset['data'][key]:
                data[key][inner_key] = dataset['data'][key][inner_key][:]
                shape_key = inner_key+"_shape"
                if not shape_key in meta_data:
                    meta_data[shape_key] = data[key][inner_key].shape
                    print(inner_key+"_shape:", meta_data[inner_key+"_shape"])
                if inner_key == 'input_story_attentive':
                    meta_data['max_len_input_story_attentive'] = data[key][inner_key].shape[1]

        self.meta_data = meta_data
        self.data = data
        f.close()
        logger.info('finish init dataset with %s' % self.data_path)


def train_option(update_dict=None, EPOCHS=50):
    BATCH_SIZE = 64
    dataset = McDataset("data.h5")
    data = dataset.data
    train_data, valid_data, test_data = data['train'], data['valid'], data['test']

    lr_cutting = LearningRateCutting(patience=1, cut_ratio=0.5)
    eval_callback = Evaluation((test_data, test_data['y_hat']), monitor='acc')
    callbacks_list = [
                      EarlyStopping(patience=3, verbose=1, monitor='val_acc'),
                      lr_cutting,
                      eval_callback,
                      ]

    model = PHM('model.yaml', dataset.meta_data, None, update_dict=update_dict)
    graph = model.build()
    graph.summary()

    try:
        logger.info('finished loading models')
        graph.fit(x=train_data, y=train_data['y_hat'],
                  validation_data=[valid_data, valid_data['y_hat']], batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=1,
                  shuffle=True, callbacks=callbacks_list
                  )
    except KeyboardInterrupt:
        logger.info('interrupted by the user, and continue to eval on test.')


if __name__ == '__main__':
    import argparse
    setup_logging(default_path='logging.yaml', default_level=logging.INFO)
    parser = argparse.ArgumentParser(description="train option model and print out results.")
    parser.add_argument("-e", "--epoch", type=int, default=10, help="number of epoch to train.")
    args = parser.parse_args()

    train_option(EPOCHS=args.epoch)
    logger.info("**************Train_eval finished******************")
