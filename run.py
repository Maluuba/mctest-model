# encoding: utf-8
import os
import logging
import numpy as np
import codecs
import shutil

from keras.callbacks import EarlyStopping
from setup_logger import setup_logging

from model import PHM
from callbacks import LearningRateCutting, Evaluation, ModelCheckpoint
import cPickle as pickle
import time
from datasets.race import PHMFeature
from save_model import save_weights

logger = logging.getLogger(__name__)


class McDataset(object):
    def __init__(self, data_path, which_dataset):
        self.data_path = data_path
        self.dataset_name = which_dataset
        self._load()

    def _load(self):
        import h5py
        print 'init dataset with h5 file.'
        meta_data = {}
        print 'data_path:', self.data_path
        # f = h5py.File(self.data_path, 'r')
        # dataset = f[self.dataset_name]

        # for key in dataset.attrs:
        #     meta_data[key] = dataset.attrs[key]

        # words_flatten = f['words_flatten'][0]
        # id2word = words_flatten.split('\n')
        # id2word =  f['vocab'][:]
        # # assert len(self.id2word) == f.attrs['vocab_len'], "%s != %s" % (len(id2word), f.attrs['vocab_len'])
        # word2id = dict(zip(id2word, range(len(id2word))))
        # meta_data['id2word'] = id2word
        # meta_data['word2id'] = word2id
        # meta_data['idfs'] = dataset['idfs'][:]

        # meta_data['answer_size'] = dataset['data/train/input_answer'].shape[1]
        # meta_data['n_s'] = dataset['data/train/input_story'].shape[1]
        # meta_data['n_voc'] = len(id2word)
        # meta_data['n_w_a'] = dataset['data/train/input_answer'].shape[-1]
        # meta_data['n_w_q'] = dataset['data/train/input_question'].shape[-1]
        # meta_data['n_w_qa'] = dataset['data/train/input_question_answer'].shape[-1]
        # meta_data['n_w_s'] = dataset['data/train/input_story'].shape[-1]
        # meta_data['w2v_path'] = '/opt/dnn/word_embedding/glove.840B.300d.pandas.hdf5'
        # meta_data['stop_words_file'] = '/home/xihlin/workspace/ExamComprehension/examcomprehension/datasets/stopwords.txt'

        with codecs.open('stopwords.txt', 'r', 'utf-8') as f:
            stopwords = [l.strip() for l in f]
        phm_data = PHMFeature(stopwords=stopwords, saved_h5=self.data_path)
        id2word =  phm_data.vocab
        word2id = dict(zip(id2word, range(len(id2word))))
        meta_data['idfs'] = phm_data.idfs(self.dataset_name)
        meta_data['id2word'] = id2word
        meta_data['word2id'] = word2id

        # dataset = getattr(phm_data, self.dataset_name)
        # meta_data['answer_size'] = dataset['train']['input_answer'].shape[1]
        # meta_data['n_s'] = dataset['train']['input_story'].shape[1]
        # meta_data['n_voc'] = len(id2word)
        # meta_data['n_w_a'] = dataset['train']['input_answer'].shape[-1]
        # meta_data['n_w_q'] = dataset['train']['input_question'].shape[-1]
        # meta_data['n_w_qa'] = dataset['train']['input_question_answer'].shape[-1]
        # meta_data['n_w_s'] = dataset['train']['input_story'].shape[-1]

        meta_data['answer_size'] = phm_data.max_n_options
        meta_data['n_s'] = phm_data.max_n_sents 
        meta_data['n_voc'] = len(id2word)
        meta_data['n_w_a'] = phm_data.max_option_len 
        meta_data['n_w_q'] = phm_data.max_question_len
        meta_data['n_w_qa'] = phm_data.max_question_option_len 
        meta_data['n_w_s'] = phm_data.max_sent_len 

        meta_data['max_len_input_story_attentive'] = phm_data.max_article_len
        for k, v in phm_data._cache['high']['data']['train'].iteritems():
            meta_data[k + '_shape'] = v.shape

        dataset = getattr(phm_data, self.dataset_name)
        if self.dataset_name != 'middle':
            dataset['middle/test'] = phm_data.middle['test']
        if self.dataset_name != 'high':
            dataset['high/test'] = phm_data.high['test']
        data = {}
        for key in dataset:  # 'train', 'valid', 'test'
            print('key=%s' % key)
            data[key] = {}
            for inner_key in dataset[key]:  # input_story, etc
                print('inner_key=%s' % inner_key)
                data[key][inner_key] = dataset[key][inner_key]
                shape_key = inner_key+"_shape"
                if not shape_key in meta_data:
                    meta_data[shape_key] = data[key][inner_key].shape
                    print(inner_key+"_shape:", meta_data[inner_key+"_shape"])
                if inner_key == 'input_story_attentive':
                    meta_data['max_len_input_story_attentive'] = data[key][inner_key].shape[1]

        print meta_data.keys()
        self.meta_data = meta_data
        self.data = data
        # f.close()
        logger.info('finish init dataset with %s' % self.data_path)


def train_option(data_path, which_dataset, model_path, update_dict=None, EPOCHS=50, BATCH_SIZE=64):
    patience = 10
    dataset = McDataset(data_path, which_dataset)
    data = dataset.data
    train_data = data.pop('train')
    valid_data = data.pop('valid')
    test_data = data

    lr_cutting = LearningRateCutting(patience=1, cut_ratio=0.8)  # 0.5
    callbacks_list = [
                      EarlyStopping(patience=patience, verbose=1, monitor='val_acc'),
                      lr_cutting,
                      ]
    for k, v in test_data.iteritems():
        eval_callback = Evaluation((v, v['y_hat']), monitor='acc', name=k)
        callbacks_list.append(eval_callback)
    checkpoint = ModelCheckpoint(os.path.join(model_path, 'model.h5'),
                                 monitor='val_acc', save_best_only=True)
    callbacks_list.append(checkpoint)

    model_yaml_path = os.path.join(model_path, 'model.yaml')
    if not os.path.exists(model_yaml_path):
        shutil.copyfile('model.yaml', model_yaml_path)
    model = PHM(model_yaml_path, dataset.meta_data, None, update_dict=update_dict)
    graph = model.build()
    # from ipdb import set_trace; set_trace()
    for i, node in enumerate(graph.get_layer('membedding_1').inbound_nodes):
        print i, node.inbound_layers[0].name
    print ''
    for i, node in enumerate(graph.get_layer('membedding_2').inbound_nodes):
        print i, node.inbound_layers[0].name
    graph.summary()

    try:
        logger.info('finished loading models')
        graph.fit(x=train_data, y=train_data['y_hat'],
                  validation_data=[valid_data, valid_data['y_hat']], batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=1,
                  shuffle=True, callbacks=callbacks_list
                  )
        # save_weights(graph, os.path.join(model_path, 'model.h5'))
    except BaseException as e:
        logger.info('interrupted by the user, and continue to eval on test.')
        # save_weights(graph, os.path.join(model_path, 'model.h5'))
        raise e


if __name__ == '__main__':
    import argparse
    setup_logging(default_path='logging.yaml', default_level=logging.INFO)
    parser = argparse.ArgumentParser(description="train option model and print out results.")
    parser.add_argument("-m", "--model-path", type=str, help="directory to save model")
    parser.add_argument("-p", "--datapath", type=str, help="path to hdf5 data")
    parser.add_argument("-d", "--dataset", type=str, default='middle', help="which dataset")
    parser.add_argument("-e", "--epoch", type=int, default=10, help="number of epoch to train.")
    parser.add_argument("-b", "--batch", type=int, default=64, help="batch size.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    train_option(args.datapath, args.dataset, args.model_path, EPOCHS=args.epoch, BATCH_SIZE=args.batch)
    logger.info("**************Train_eval finished******************")
