from __future__ import print_function
import os
import json
import codecs
import numpy
from collections import Counter
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
import networkx as nx
import tempfile
import subprocess
import re
from itertools import izip
from joblib import cpu_count, delayed, Parallel
import h5py

RACE_DATA_PATH = '/opt/dnn/data/race'  # raw data is saved on Azure matchine malu25
END_OF_ARTICLE = '<END-OF-ARTICLE>'


def load_data():
    return RACE(RACE_DATA_PATH).all_data


class RACE(object):
    '''Read RACE data, indexing and dump as hdf5 which can be used for training directly.
       This class only gives article, question, option token index, no engineered feature involved.'''
    def __init__(self, path, max_article_len=1391, max_question_len=76, max_option_len=115, right_padding=True):
        self.path = path
        self.max_article_len = max_article_len
        self.max_question_len = max_question_len
        self.max_option_len = max_option_len
        self.right_padding = right_padding
        self.vocab = ['<pad>', '<unk>']
        self._cache = {}

    @staticmethod
    def combine_data(*data):
        # return tuple([a for di in d for a in di] for d in zip(*data))
        tuple([a for di in d for a in di] for d in zip(*data))

    @property
    def middle(self):
        if 'middle' not in self._cache:
            train = self.read_and_pad_a_subset('train/middle', update_vocab=True)
            dev = self.read_and_pad_a_subset('dev/middle')
            high = self.read_and_pad_a_subset('test/middle')
            self._cache['middle'] = (train, dev, high)
        return self._cache['middle']

    @property
    def high(self):
        if 'high' not in self._cache:
            train = self.read_and_pad_a_subset('train/high', update_vocab=True)
            dev = self.read_and_pad_a_subset('dev/high')
            high = self.read_and_pad_a_subset('test/high')
            self._cache['high'] = (train, dev, high)
        return self._cache['high']

    @property
    def all_data(self):
        middle = self.middle
        high = self.high
        train = self.combine_data(middle[0], high[0])
        dev = self.combine_data(middle[1], high[1])
        test = self.combine_data(middle[2], high[2])
        return train, dev, test

    def dump(self, filepath, format=None):
        answer_map = {c: i for i, c in enumerate('ABCD')}
        if format is None:
            ext = os.path.splitext(filepath)[1][1:]
            format = ext
        assert format in ['pkl', 'h5', 'npy']
        types = ['middle'] * 3 + ['high'] * 3
        splitnames = ['train', 'dev', 'test'] * 2
        # setnames= ['articles', 'articles_indices', 'questions', 'options', 'answers']
        setnames= ['articles', 'questions', 'options', 'answers']
        data = {}
        for t, s, dat in zip(types, splitnames, self.middle + self.high):
            for sn, d in zip(setnames, dat):
                data['/'.join([s, t, sn])] = d
        if format == 'pkl':
            import cPickle as pickle
            with open(filepath, 'w') as f:
                pickle.dump(f, data)
        elif format == 'npy':
            numpy.save(filepath, data)
        else:
            h5 = h5py.File(filepath, 'w')
            try:
                for k, v in data.iteritems():
                    if k.endswith('answers'):
                        v = [answer_map[a] for a in v]
                    h5[k] = v
                vocab = [a.encode('utf-8') for a in self.vocab]
                dtype = h5py.special_dtype(vlen=unicode)
                h5.create_dataset('vocab', data=vocab, dtype=dtype)
            except BaseException as e:
                h5.close()
                raise e
            h5.close()

    def read_and_pad_a_subset(self, dataset, update_vocab=False):
        data = self.read_a_dir(dataset, update_vocab)
        return self.pad_a_dataset(data, self.right_padding)

    def pad_a_dataset(self, dataset, right_padding=True):
        from keras.preprocessing.sequence import pad_sequences
        # articles, articles_indices, questions, options, answers = dataset
        articles, questions, options, answers = dataset
        mode = 'post' if self.right_padding else 'pre'
        articles = pad_sequences(articles, self.max_article_len,
                                 padding=mode, truncating=mode)
        questions = pad_sequences(questions, self.max_question_len,
                                  padding=mode, truncating=mode)
        options = [pad_sequences(opt, self.max_option_len,
                                 padding=mode, truncating=mode)
                   for opt in options]
        options = numpy.asarray(options)
        answers = numpy.asarray(answers)
        # return articles, articles_indices, questions, options, answers
        return articles, questions, options, answers

    def read_a_dir(self, dataset, update_vocab=False):
        '''Read a dataset, e.g., race/train/high from raw text files'''
        print('Reading %s' % dataset)
        dataset_path = os.path.join(RACE_DATA_PATH, dataset)
        answer_map = {c: i for i, c in enumerate('ABCD')}
        article_ids, articles, articles_indices = [], [], []
        questions, options, answers = [], [], []
        article_idx = 0
        for fl in tqdm(os.listdir(dataset_path)):
            if not fl.endswith('.txt'):
                continue
            path = os.path.join(dataset_path, fl)
            with codecs.open(path, 'r', 'utf-8') as f:
                for l in f:
                    smp = json.loads(l.strip())
                    s = word_tokenize(smp['article'].lower())
                    for i, (q, opts, a) in enumerate(zip(smp['questions'], smp['options'], smp['answers'])):
                        questions.append(word_tokenize(q.lower()))
                        options.append([word_tokenize(opt.lower()) for opt in opts])
                        answers.append(answer_map[a])
                        articles_indices.append(article_idx)
                        article_ids.append(smp['id'])
                        articles.append(s)
                    article_idx += 1

        word_counts = Counter(w for t in articles for w in t)
        word_counts.update(w for t in questions for w in t)
        word_counts.update(w for t in options for o in q for w in o)

        if update_vocab:
            self.vocab += list(set(word_counts.keys()) - set(self.vocab))
            self.word2id = {w: i for i, w in enumerate(self.vocab)}

        articles = [[self.word2id.get(w, 1) for w in t] for t in articles]
        questions = [[self.word2id.get(w, 1) for w in t] for t in questions]
        options = [[[self.word2id.get(w, 1) for w in o] for o in t] for t in options]

        # return articles, articles_indices, questions, options, answers
        return articles, questions, options, answers


class PHMFeature(object):
    '''
    Read RACE data, indexing and dump as hdf5 which can be used for training directly.
    This classs includes a bunch of dependency parsing and sentence-ngram types of engineered features

    # Data overview (of MCTest)
    input_answer <HDF5 dataset "input_answer": shape (1000, 4, 24), type "<i4">
    input_dep <HDF5 dataset "input_dep": shape (1000, 51, 4), type "<f4">
    input_dep_2gram <HDF5 dataset "input_dep_2gram": shape (1000, 50, 4), type "<f4">
    input_dep_3gram <HDF5 dataset "input_dep_3gram": shape (1000, 49, 4), type "<f4">
    input_negation_questions <HDF5 dataset "input_negation_questions": shape (1000, 4), type "<f4">
    input_question <HDF5 dataset "input_question": shape (1000, 19), type "<i4">
    input_question_answer <HDF5 dataset "input_question_answer": shape (1000, 4, 26), type "<i4">
    input_reordered_story <HDF5 dataset "input_reordered_story": shape (1000, 51, 83), type "<i4">
    input_reordered_story_2gram <HDF5 dataset "input_reordered_story_2gram": shape (1000, 50, 98), type "<i4">
    input_reordered_story_3gram <HDF5 dataset "input_reordered_story_3gram": shape (1000, 49, 141), type "<i4">
    input_reordered_story_attentive <HDF5 dataset "input_reordered_story_attentive": shape (1000, 632), type "<i4">
    input_story <HDF5 dataset "input_story": shape (1000, 51, 83), type "<i4">
    input_story_2gram <HDF5 dataset "input_story_2gram": shape (1000, 50, 98), type "<i4">
    input_story_3gram <HDF5 dataset "input_story_3gram": shape (1000, 49, 141), type "<i4">
    input_story_attentive <HDF5 dataset "input_story_attentive": shape (1000, 632), type "<i4">
    y_hat <HDF5 dataset "y_hat": shape (1000, 4), type "<i4">
    '''

    def __init__(self, max_n_sents=50, max_sent_len=50, max_question_len=50, max_option_len=50,
                 right_padding=True, stopwords=[], saved_h5=None):
        self.basic_features = ['input_story', 'input_dep', 'input_reordered_story',
                               'input_question', 'input_answer', 'y_hat']
        self._cache = {}
        self.max_n_sents = max_n_sents
        self.max_sent_len = max_sent_len
        self.max_question_len = max_question_len
        self.max_option_len = max_option_len
        self.max_n_options = 4
        self.max_question_option_len = int((max_option_len + max_question_len) * 0.75)
        self.max_article_len = 1000
        self.right_padding = right_padding
        self.vocab = ['<pad>', '<unk>']
        if saved_h5 is not None:
            self._construct_from_h5(saved_h5)
        self.stopwords = set(stopwords)

    def _construct_from_h5(self, data_path):
        self._cache = self._read_from_h5(data_path)
        self.vocab = self._cache.pop('vocab').tolist()
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        if 'middle' in self._cache:
            dataset = self._cache['middle']
        elif 'high' in self._cache:
            dataset = self._cache['high']
        else:
            raise Exception('Invalid hdf5 data')
        self.max_n_sents = dataset['data']['train']['input_story'].shape[1]
        self.max_sent_len = dataset['data']['train']['input_story'].shape[-1]
        self.max_question_len = dataset['data']['train']['input_question'].shape[-1]
        self.max_option_len = dataset['data']['train']['input_answer'].shape[-1]
        self.max_n_options = dataset['data']['train']['input_answer'].shape[1]
        if 'input_question_answer' in dataset['data']['train']:
            self.max_question_option_len = dataset['data']['train']['input_question_answer'].shape[-1]
        if 'input_story_attentive' in dataset['data']['train']:
            self.max_article_len = dataset['data']['train']['input_story_attentive'].shape[-1]

    def _add_infered_features(self, data):
        options = data['input_answer']
        articles = data['input_story']
        questions = data['input_question']
        reordered = data['input_reordered_story']
        dep_scores = data['input_dep']
        negation_indices = []
        for w in ['no', 'not']:
            if w in self.word2id:
                negation_indices.append(self.word2id[w])
        # remove paddings
        articles = [[[w for w in s if w > 0] for s in t if any(s > 0)] for t in articles]
        questions = [[w for w in q if w > 0] for q in questions]
        options = [[[w for w in o if w > 0] for o in t if any(o > 0)] for t in options]
        reordered = [[[w for w in s if w > 0] for s in t if any(s > 0)] for t in reordered]
        # dep_scores = [[[w for w in o] for o in t] for t in dep_scores]

        negation = (-1 if any(w in negation_indices for w in q) else 1 for q in questions)
        infered_features = {}
        infered_features['input_negation_questions'] = [[q for _ in opts] for q, opts in izip(negation, options)]
        infered_features['input_question_answer'] = [[q + o for o in opts] for q, opts in izip(questions, options)]
        infered_features['input_story_2gram'] = [[s1 + s2 for s1, s2 in izip(t[:-1], t[1:])] for t in articles]
        infered_features['input_story_3gram'] = [[s1 + s2 + s3 for s1, s2, s3 in izip(t[:-2], t[1:-1], t[2:])] for t in articles]
        infered_features['input_reordered_story_2gram'] = [[s1 + s2 for s1, s2 in izip(t[:-1], t[1:])] for t in reordered]
        infered_features['input_reordered_story_3gram'] = [[s1 + s2 + s3 for s1, s2, s3 in izip(t[:-2], t[1:-1], t[2:])] for t in reordered]
        infered_features['input_story_attentive'] = [[w for s in t for w in s if w > 0] for t in articles]  # list of words (flatten sentences)
        infered_features['input_reordered_story_attentive'] = [[w for s in t for w in s if w > 0] for t in reordered]
        infered_features['input_dep_2gram'] = [[numpy.asarray(s1) + numpy.asarray(s2) for s1, s2 in izip(ds[:-1], ds[1:])] for ds in dep_scores]
        infered_features['input_dep_3gram'] = [[numpy.asarray(s1) + numpy.asarray(s2) + numpy.asarray(s3)
                                                for s1, s2, s3 in izip(ds[:-2], ds[1:-1], ds[2:])]
                                               for ds in dep_scores]
        infered_features = self.pad_a_dataset(infered_features)
        for k, v in infered_features.iteritems():
            data[k] = v

    @staticmethod
    def _read_from_h5(data_path):
        def _visit(data):
            if isinstance(data, h5py.Dataset):
                return data[:]
            else:
                return {k: _visit(v) for k, v in data.iteritems()}
        h5 = h5py.File(data_path, 'r')
        try:
            output = _visit(h5)
            h5.close()
        except BaseException as e:
            h5.close()
            raise e
        return output

    def getdata(self, which, include_infered_features=False):
        if not self._cache.get(which):
            self._cache[which] = {'data': {}}
            data = self._cache[which]['data']
            try:
                data['train'] = self.read_a_dir('train/%s' % which, True)
                data['train'] = self.pad_a_dataset(data['train'])
                data['valid'] = self.read_a_dir('dev/%s' % which, False)
                data['valid'] = self.pad_a_dataset(data['valid'])
                data['test'] = self.read_a_dir('test/%s' % which, False)
                data['test'] = self.pad_a_dataset(data['test'])
            except BaseException as e:
                del self._cache[which]
                raise e
        return self._cache[which]['data']

    @property
    def middle(self):
        if 'middle' not in self._cache:
            self.getdata('middle')
        if 'input_question_answer' not in self._cache['middle']['data']['train']:
            for k, v in self._cache['middle']['data'].iteritems():
                print('middle/%s' % k)
                self._add_infered_features(v)
        return self._cache['middle']['data']

    @property
    def high(self):
        if 'high' not in self._cache:
            self.getdata('high')
        if 'input_question_answer' not in self._cache['high']['data']['train']:
            for k, v in self._cache['high']['data'].iteritems():
                print('high/%s' % k)
                self._add_infered_features(v)
        return self._cache['high']['data']

    @staticmethod
    def combine_data(middle, high):
        output = {}
        for k in middle:
            output[k] = numpy.concatenate([middle[k], high[k]], axis=0)
        return output

    @property
    def both(self):
        middle = self.middle
        high = self.high
        output = {}
        for k in middle:
            output[k] = self.combine_data(middle[k], high[k])
        return output

    def idfs(self, which=None):
        if which == 'middle':
            idfs = self.get_doc_counts('middle')
        elif which == 'high':
            idfs = self.get_doc_counts('middle')
        else:
            idfs_m = self.get_doc_counts('middle')
            idfs_h = self.get_doc_counts('high')
            idfs = idfs_m + idfs_h
        idfs = numpy.log((idfs.sum() + 0.5) / (idfs + 0.5))
        idfs /= idfs.max()
        idfs[:2] = 0.  # <pad>, <unk>
        return idfs

    def get_doc_counts(self, which):
        data = self.getdata(which)
        doc_counts = numpy.zeros((len(self.vocab), 1), dtype='float32')
        for p in data['train']['input_story']:
            wordset = list(set(p.ravel()))
            doc_counts[wordset] += 1
        return doc_counts

    def dump(self, outfile, which=None):
        import h5py
        if which in [None, 'both']:
            self.dump(outfile, 'middle')
            self.dump(outfile, 'high')
        else:
            assert which in ('middle', 'high')
            data = self.getdata(which)
            idfs = self.idfs(which)

            h5 = h5py.File(outfile, 'a')
            try:
                vocab = [a.encode('utf-8') for a in self.vocab]
                dtype = h5py.special_dtype(vlen=unicode)
                if 'vocab' in h5:
                    del h5['vocab']
                h5.create_dataset('vocab', data=vocab, dtype=dtype)
                idf_key = '%s/idfs' % which
                if idf_key in h5:
                    del h5[idf_key]
                h5[idf_key] = idfs.reshape((-1, 1))
                h5.create_group('%s/data' % which)
                h5_data = h5['%s/data' % which]
                for k, v in data.iteritems():
                    for k2, v2 in v.iteritems():
                        if k2 not in self.basic_features:
                            continue
                        print('%s/%s' % (k, k2))
                        h5_data['%s/%s' % (k, k2)] = v2
                h5.close()
            except BaseException as e:
                h5.close()
                raise e

    def read_a_dir(self, dataset, update_vocab=False):
        '''Read a dataset, e.g., race/train/high from raw text files'''
        print('Reading %s' % dataset)
        dataset_path = os.path.join(RACE_DATA_PATH, dataset)
        articles = []
        questions, options, answers = [], [], []
        reordered, dep_scores = [], []
        deps = read_batch_parser_result(os.path.join(dataset_path, 'all.article.dep'), os.path.join(dataset_path, 'all.id'))
        files = [os.path.join(dataset_path, fl) for fl in os.listdir(dataset_path) if fl.endswith('.txt')]
        samples = []
        for fl in files:
            with codecs.open(fl, 'r', 'utf-8') as f:
                smp = json.load(f)
            samples.append(smp)
        deps = [deps[smp['id']] for smp in samples]

        def parallel_map(func, iterator_of_args, n_jobs=None):
            if n_jobs is None or n_jobs <= 0:
                n_jobs = max(1, cpu_count() - 1)
            par = Parallel(n_jobs=n_jobs)
            return par(delayed(func)(*x) for x in iterator_of_args)

        results = parallel_map(process_an_article, izip(tqdm(samples), deps))
        reducer = lambda x, y: [xi + yi for xi, yi in izip(x, y)]
        articles_ids, articles, reordered, dep_scores, questions, options, answers = reduce(reducer, results)

        word_counts = Counter(w for t in articles for s in t for w in s)
        word_counts.update(w for t in questions for w in t)
        word_counts.update(w for t in options for o in t for w in o)

        if update_vocab:
            self.vocab += list(set(word_counts.keys()) - set(self.vocab) - set(self.stopwords))
            self.word2id = {w: i for i, w in enumerate(self.vocab)}

        articles = [[[self.word2id.get(w, 1) for w in s if w not in self.stopwords] for s in t] for t in articles]
        questions = [[self.word2id.get(w, 1) for w in t if w not in self.stopwords] for t in questions]
        options = [[[self.word2id.get(w, 1) for w in o if w not in self.stopwords] for o in t] for t in options]
        reordered = [[[self.word2id.get(w, 1) for w in s if w not in self.stopwords] for s in t] for t in reordered]

        output = {}
        output['input_answer'] = options
        output['input_story'] = articles
        output['input_question'] = questions
        output['input_reordered_story'] = reordered
        output['input_dep'] = dep_scores

        '''
        negation_indices = []
        for w in ['no', 'not']:
            if w in self.word2id:
                negation_indices.append(self.word2id[w])
        negation = (-1 if any(w in negation_indices for w in q) else 1 for q in questions)
        output['input_negation_questions'] = [[q for _ in opts] for q, opts in izip(negation, options)]
        output['input_question_answer'] = [[q + o for o in opts] for q, opts in izip(questions, options)]
        output['input_story_2gram'] = [[s1 + s2 for s1, s2 in izip(t[:-1], t[1:])] for t in articles]
        output['input_story_3gram'] = [[s1 + s2 + s3 for s1, s2, s3 in izip(t[:-2], t[1:-1], t[2:])] for t in articles]
        output['input_reordered_story_2gram'] = [[s1 + s2 for s1, s2 in izip(t[:-1], t[1:])] for t in reordered]
        output['input_reordered_story_3gram'] = [[s1 + s2 + s3 for s1, s2, s3 in izip(t[:-2], t[1:-1], t[2:])] for t in reordered]
        output['input_story_attentive'] = [[w for s in t for w in s if w > 0] for t in articles]  # list of words (flatten sentences)
        output['input_reordered_story_attentive'] = [[w for s in t for w in s if w > 0] for t in reordered]
        output['input_dep_2gram'] = [[numpy.asarray(s1) + numpy.asarray(s2) for s1, s2 in izip(ds[:-1], ds[1:])] for ds in dep_scores]
        output['input_dep_3gram'] = [[numpy.asarray(s1) + numpy.asarray(s2) + numpy.asarray(s3)
                                     for s1, s2, s3 in izip(ds[:-2], ds[1:-1], ds[2:])]
                                     for ds in dep_scores]
        '''

        max_n_options = max(len(opts) for opts in options)
        assert max_n_options == self.max_n_options
        output['y_hat'] = numpy.eye(max_n_options)[answers]
        return output

    def pad_a_dataset(self, output):
        from keras.preprocessing.sequence import pad_sequences
        expected_shapes = {'input_answer': (self.max_n_options, self.max_option_len),
                           'input_dep': (self.max_n_sents, self.max_n_options),
                           'input_dep_2gram': (self.max_n_sents - 1, self.max_n_options),
                           'input_dep_3gram': (self.max_n_sents - 2, self.max_n_options),
                           'input_negation_questions': (self.max_n_options,),
                           'input_question': (self.max_question_len,),
                           'input_question_answer': (self.max_n_options, self.max_question_option_len),
                           'input_reordered_story': (self.max_n_sents, self.max_sent_len),
                           'input_reordered_story_2gram': (self.max_n_sents - 1, int(self.max_sent_len * 1.5)),
                           'input_reordered_story_3gram': (self.max_n_sents - 2, self.max_sent_len * 2),
                           'input_reordered_story_attentive': (self.max_article_len,),
                           'input_story': (self.max_n_sents, self.max_sent_len),
                           'input_story_2gram': (self.max_n_sents - 1, int(self.max_sent_len * 1.5)),
                           'input_story_3gram': (self.max_n_sents - 2, self.max_sent_len * 2),
                           'input_story_attentive': (self.max_article_len,),
                           }

        mode = 'post'
        for k, shape in expected_shapes.iteritems():
            if k not in output:
                continue
            v = output[k]
            if len(shape) == 2:
                assert type(v[0][0]) in [list, tuple, numpy.ndarray]
                res = []
                for vi in v:
                    r = pad_sequences(vi, shape[1], padding=mode, truncating=mode)
                    if len(r) >= shape[0]:
                        r = r[:shape[0]]
                    else:
                        p = numpy.zeros((shape[0] - len(r), shape[1]), dtype=r.dtype)
                        r = numpy.concatenate([r, p], axis=0)
                    res.append(r)
                output[k] = numpy.asarray(res)
            elif len(shape) == 1:
                assert type(v[0][0]) not in [list, tuple]
                r = pad_sequences(v, shape[0], padding=mode, truncating=mode)
                output[k] = r
            else:
                raise
        return output


def prepare_sentences_for_stanford_parser(dataset, output_file):
    dataset_path = os.path.join(RACE_DATA_PATH, dataset)
    output_id_file = output_file + '.id'
    with codecs.open(output_file, 'w', 'utf-8') as fo, codecs.open(output_id_file, 'w', 'utf-8') as fid:
        for fl in tqdm(os.listdir(dataset_path)):
            if not fl.endswith('.txt'):
                continue
            article_ids, articles = [], []
            path = os.path.join(dataset_path, fl)
            with codecs.open(path, 'r', 'utf-8') as f:
                for l in f:
                    smp = json.loads(l.strip())
                    # s = [word_tokenize(si) for si in sent_tokenize(smp['article'])]
                    s = sent_tokenize(smp['article'])
                    for si in s:
                        articles.append(si)
                        article_ids.append(smp['id'])
            # fo.write('\n'.join(' '.join(si) for si in articles) + '\n')
            fo.write('\n'.join(articles) + '\n')
            fid.write('\n'.join(article_ids) + '\n')


def parse_dep_arc(sentence):
    '''
    u'case(holiday-5, During-1)' -> (u'case', u'holiday-5', u'During-1', u'during')
    where is the last item is the word in the sentence (lower cased)
    '''
    res = re.search(r'^(.*?)\((.*?-\d+), ((.*?)-\d+)\)$', sentence).groups()
    res = [w.lower() for w in res]
    return res


def read_dep_file(dep_file):
    with codecs.open(dep_file, 'r', 'utf-8') as f:
        dep = [a.split('\n') for a in f.read().split('\n\n') if a.strip()]
    dep = [[parse_dep_arc(item) for item in s] for s in dep]
    return dep


def stanford_dep_parser(infile, outfile, stanford_parser_path='~/tmp/stanford-parser-full-2016-10-31'):
    '''
    :param infile: text file normally passed to stanford parser
    :param outfile: dependency text in arc format
    '''
    cmd = 'cd %s; \
          java -Xmx2g -cp "*" edu.stanford.nlp.parser.nndep.DependencyParser \
          -model edu/stanford/nlp/models/parser/nndep/english_UD.gz \
          -textFile %s \
          -outFile %s &> /dev/null' % (stanford_parser_path, infile, outfile)
    subprocess.check_call(cmd, shell=True)
    # status = os.system(cmd)
    # if status != 0:
    #     OSError("Return status %d,  when calling %s" % (status, cmd))


def _parse_dep_to_file(article, output_file=None, stanford_parser_path='~/tmp/stanford-parser-full-2016-10-31'):
    article_file ='%s/dep_%d.article' % (tempfile.tempdir, hash(article))
    if output_file is None:
        dep_file = '%.dep' % article_file
    else:
        dep_file = output_file
    with codecs.open(article_file, 'w', 'utf-8') as f:
        f.write(article)
    try:
        stanford_dep_parser(article_file, dep_file, stanford_parser_path)
    except OSError as e:
        os.remove(article_file)
        raise e
    os.remove(article_file)
    return dep_file


def parse_dep(article):
    dep_file = _parse_dep_to_file(article)
    dep = read_dep_file(dep_file)
    os.remove('%s.dep' % article)
    return dep


def parse_dep_file(json_file):
    with codecs.open(json_file, 'r', 'utf-8') as f:
        article = json.load(f)['article']
    dep_file = '%s.article.dep' % json_file
    return _parse_dep_to_file(article, dep_file)


def parse_dep_dir(dataset_dir):
    '''parse all files in a directory'''
    for fl in tqdm(os.listdir(dataset_dir)):
        if not fl.endswith('.txt'):
            continue
        path = os.path.join(dataset_dir, fl)
        parse_dep_file(path)


def get_reordered_sent(sent_dep):
    sent = [item[3] for item in sent_dep]
    g_dep = nx.Graph()
    for e in sent_dep:
        g_dep.add_edge(*e[1:3])
    nodes = g_dep.nodes()
    try:
        fieldler_vec = nx.fiedler_vector(g_dep, method='lobpcg')
    except nx.NetworkXError as err1:
        print('1:', err1)
        try:
            if len(sent_dep) > 2:
                g_dep = nx.Graph()
                for e in sent_dep[:-1]:
                    g_dep.add_edge(*e[1:3])
                fieldler_vec = nx.fiedler_vector(g_dep, method='lobpcg')
                nodes = g_dep.nodes()
                sent = sent[:-1]
        except nx.NetworkXError as err2:
            print('2:', err2)
            fieldler_vec = numpy.zeros((len(nodes),))
    reordered = [re.findall(r'^(.*?)-\d+$', nodes[i])[0]
                 for i in fieldler_vec.argsort() if nodes[i] != 'root-0']
    return sent, reordered


def get_dependency_scores(sent_dep, options):
    '''
    :param setnt_dep: list of list(len=4), i.e., result from `parse_dep_arc`
    :param options: list (len=4) of list of tokens
    :return reordered sentence, option-to-sentence dependence scores (size of max connected subtree)
    '''
    g_dep = nx.Graph()
    for e in sent_dep:
        g_dep.add_edge(*e[1:3])
    nodes = g_dep.nodes()
    dep_scores = []
    for opt in options:
        sub = set(w.lower() for w in opt)
        sub = [x for x in nodes if re.findall(r'^(.*?)-\d+$', x)[0].lower() in sub]
        if sub:
            g_dep_sub = g_dep.subgraph(sub)
            dep_sub_max_con = max(nx.connected_components(g_dep_sub), key=lambda x: len(x))
            max_con_len = len(dep_sub_max_con)
            score = float(max_con_len) / len(nodes)
        else:
            score = 0.
        dep_scores.append(score)
    return dep_scores


def gather_all_articles(dataset_dir):
    '''collect all articles in a directoris and saved in to file, which article separator `END_OF_ARTICLE`
       This is the file used for the Stanford dependency parser, which can be called by `stanford_dep_parser`
    '''
    id_file = os.path.join(dataset_dir, 'all.id')
    article_file = os.path.join(dataset_dir, 'all.article')
    with codecs.open(id_file, 'w', 'utf-8') as fid, codecs.open(article_file, 'w', 'utf-8') as fo:
        for fl in tqdm(os.listdir(dataset_dir)):
            if not fl.endswith('.txt'):
                continue
            fl = os.path.join(dataset_dir, fl)
            with codecs.open(fl, 'r', 'utf-8') as f:
                data = json.load(f)
                fid.write(data['id'] + '\n')
                fo.write(data['article'] + ('\n . %s.\n' % END_OF_ARTICLE))


def read_batch_parser_result(dep_file, id_file):
    '''Read Stanford dependency paser outputs
    '''
    dep = []
    with codecs.open(dep_file, 'r', 'utf-8') as f:
        raw = f.read().strip().split('\n\n')
    eoa_lines = [i for i, s in enumerate(raw) if END_OF_ARTICLE in s]
    j = 0
    dep_an_article = []
    for i, sent in enumerate(raw):
        if END_OF_ARTICLE in sent:
            j += 1
            dep.append(dep_an_article)
            dep_an_article = []
            continue
        sent = sent.split('\n')
        p = [parse_dep_arc(item) for item in sent]
        if i == eoa_lines[j] - 1:
            if p[-1][-1] == '.' and all(p[-1][2] != pi[1] for pi in p[:-1]):
                # does not result in disconnetivity
                p = p[:-1]
        dep_an_article.append(p)
    with codecs.open(id_file, 'r', 'utf-8') as f:
        ids = [l.strip() for l in f]
    assert len(ids) == len(dep)
    return dict(zip(ids, dep))


def process_an_article(smp, dep):
    answer_map = {c: i for i, c in enumerate('ABCD')}
    sents, reordered_sents = zip(*[get_reordered_sent(sd) for sd in dep])
    article_ids, articles = [], []
    questions, options, answers = [], [], []
    reordered, dep_scores = [], []
    for i, (q, opts, a) in enumerate(izip(smp['questions'], smp['options'], smp['answers'])):
        score = [get_dependency_scores(sd, opts) for sd in dep]
        questions.append(word_tokenize(q.lower()))
        options.append([word_tokenize(opt.lower()) for opt in opts])
        answers.append(answer_map[a])
        article_ids.append(smp['id'])
        articles.append(sents)
        reordered.append(reordered_sents)
        dep_scores.append(score)
    return article_ids, articles, reordered, dep_scores, questions, options, answers


if __name__ == '__test__':
    '''Example workflow of gettting the Stanford dependency parser results'''
    dataset = 'train/middle'
    dataset_dir = os.path.join(RACE_DATA_PATH, dataset)
    gather_all_articles(dataset_dir)

    stanford_dep_parser(os.path.join(dataset_dir, 'all.article'),
                        os.path.join(dataset_dir, 'all.article.dep'),
                        stanford_parser_path='~/tmp/stanford-parser-full-2016-10-31')
    deps = read_batch_parser_result(os.path.join(dataset_dir, 'all.article.dep'))
