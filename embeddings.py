from keras.layers.embeddings import Layer, Embedding
from itertools import izip
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MEmbedding(Embedding):
    def get_output_shape_for(self, input_shape):
        """
        Fix out_shape so that it doesn't depend on input_length.
        :param input_shape:
        :return:
        """
        return input_shape + (self.output_dim, )

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'input_length': self.input_length,
                  'mask_zero': self.mask_zero,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'dropout': self.dropout}
        base_config = super(MEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class H5EmbeddingManager(object):
    def __init__(self, h5_path, mode='disk'):
        self.mode = mode
        import h5py
        f = h5py.File(h5_path, 'r')
        if mode == 'disk':
            self.W = f['embedding']
        elif mode == 'in-memory':
            self.W = f['embedding'][:]
        message = "load mode=%s, embedding data type=%s, shape=%s" % (self.mode, type(self.W), self.W.shape)
        logger.info(message)
        words_flatten = f['words_flatten'][0]
        self.id2word = words_flatten.split('\n')
        assert len(self.id2word) == f.attrs['vocab_len'], "%s != %s" % (len(self.id2word), f.attrs['vocab_len'])
        self.word2id = dict(izip(self.id2word, range(len(self.id2word))))
        del words_flatten

    def __getitem__(self, item):
        item_type = type(item)
        if item_type is str:
            index = self.word2id[item]
            embs = self.W[index]
            return embs
        else:
            raise RuntimeError("don't support type: %s" % type(item))

    def init_word_embedding(self, words, dim_size=300, scale=0.1, mode='google'):
        print('loading word embedding.')
        word2id = self.word2id
        W = self.W
        shape = (len(words), dim_size)
        np.random.seed(len(words))
        # W2V = np.random.uniform(low=-scale, high=scale, size=shape).astype('float32')
        W2V = np.zeros(shape, dtype='float32')
        for i, word in enumerate(words[1:], 1):
            if word in word2id:
                _id = word2id[word]
                vec = W[_id]
                vec /= np.linalg.norm(vec)
            elif word.capitalize() in word2id:
                _id = word2id[word.capitalize()]
                vec = W[_id]
                vec /= np.linalg.norm(vec)
            else:
                vec = np.random.normal(0, 1.0, 300)
                vec = (0.01 * vec).astype('float32')
            W2V[i] = vec[:dim_size]
        return W2V

    def init_word_embedding1(self, words, dim_size=300, scale=0.1, mode='google'):
        word2id = self.word2id
        W = self.W
        shape = (len(words), dim_size)
        np.random.seed(len(words))
        # W2V = np.random.uniform(low=-scale, high=scale, size=shape).astype('float32')
        W2V = np.random.normal(0, 1.0, size=shape).astype('float32') * 0.01
        W2V[0, :] = 0
        if mode == 'random':
            return W2V
        in_vocab = np.ones(shape[0], dtype=np.bool)
        oov_set = set()
        word_ids = []
        for i, word in enumerate(words):
            _id = -1
            try:
                _id = word2id[word]
            except KeyError:
                pass
            if _id < 0:
                try:
                    _id = word2id[word.capitalize()]
                except KeyError:
                    pass
            if _id < 0:
                in_vocab[i] = False
                if not word.startswith("$oov-"):
                    oov_set.update([word])
            else:
                word_ids.append(_id)
        if self.mode == 'in-memory':
            W2V[in_vocab][:, :] = W[np.array(word_ids, dtype='int32')][:, :dim_size]
        else:
            nonzero_ids = in_vocab.nonzero()[0]
            for i in nonzero_ids:
                emb = W[word_ids[i]]
                W2V[i][:] = emb[:dim_size]
        return W2V
