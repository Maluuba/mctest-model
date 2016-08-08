# encoding: utf-8
from __future__ import absolute_import
from __future__ import print_function
import logging

import yaml
import theano
import numpy as np
from keras.optimizers import Adam, Adadelta, Adagrad, RMSprop, SGD
from keras.regularizers import l2
from keras.layers import Input, RepeatVector, Dense, Dropout, merge, Activation
from keras.activations import relu
from keras import backend as K
from keras.engine import Model

from embeddings import MEmbedding, H5EmbeddingManager
from layers import TimeDistributedMerge, EmbeddingWeighting, UnitNormalization, NTimeDistributed
from constraints import PadStayZeroNonNeg, PadStayZero
from layers import WordByWordMatrix, WordByWordScores, GeneralizedMean, WordByWordSlideSumInsideSentence, \
    SlideSum, DependencyDistanceScore, Combination, WordByWordSlideSum, WordByWordSlideSumInsideSentence, TopNWordByWord


logger = logging.getLogger(__name__)
floatX = theano.config.floatX

optimizer_MAP = dict(adam=Adam, adadelta=Adadelta, adagrad=Adagrad, rmsprop=RMSprop, sgd=SGD)


def dict_update(input_dict, update_dict):
    def _update(dict1, dict2):
        if dict2 and dict1:
            for key, value in dict2.iteritems():
                if value is None or value in ["None", "none"]:
                    continue
                if key in dict1:
                    if type(value) == dict:
                        _update(dict1[key], value)
                    else:
                        dict1[key] = value
                else:
                    dict1[key] = value

    _update(input_dict, update_dict)


class PHM(object):
    def __init__(self, model_yaml, data_options, not_vector, update_dict=None, **kwargs):
        """
        :param data_options: options for data.
        :param update_dict: used to update parameter in the yaml file, for like hyper opt.
        :param kwargs:
        :return:
        """
        super(PHM, self).__init__(**kwargs)
        self.data_options = data_options
        if type(model_yaml) is dict:
            self.model_config = model_yaml
        else:
            with open(model_yaml) as reader:
                self.model_config = yaml.safe_load(reader)

        if update_dict is not None:
            dict_update(self.model_config, update_dict)

        # data config
        self.story_maxlen = data_options['max_len_input_story_attentive']
        self.max_num_sentence = data_options['n_s']
        self.query_maxlen = data_options['n_w_q']
        self.story_sentence_maxlen = data_options['n_w_s']
        self.vocab_size = data_options['n_voc']
        self.answer_size = data_options['answer_size']
        self.answer_maxlen = data_options['n_w_a']
        self.qa_maxlen = data_options['n_w_qa']

        # model config
        self.regularizer_alpha = 0.0
        self.embed_type = self.model_config['layers']['embedding']['type']
        self.EMBED_SIZE = self.model_config['layers']['embedding']['embedding_size']
        if self.embed_type == 'random':
            self.W2V = None
        else:
            embedding = H5EmbeddingManager('GoogleNews-vectors-negative300.h5', mode='in-memory')
            self.W2V = embedding.init_word_embedding(self.data_options['id2word'], dim_size=self.EMBED_SIZE,
                                                     mode=self.embed_type)
            del embedding
            self.W2V = [self.W2V]

        self.model_name = "PHM"
        self.inputs_nodes = dict()
        self.nodes = dict()

    def get_predict_fun(self):
        if self.graph.predict_function is None:
            self.graph._make_predict_function()
        return self.graph.predict_function.function

    def save_model(self, predict_path="predict.pkl"):
        with open(predict_path, 'wb') as f:
            import sys
            logger.info('saving pre_function.')
            sys.setrecursionlimit(10000)
            import cPickle as pkl
            pkl.dump(self.get_predict_fun(), f, -1)

    def _init_word_weight(self, init_type=None):
        weight_emb_size = 1
        options = self.data_options
        w2idx, idx2w = options['word2id'], options['id2word']
        voc_size = len(w2idx)
        init_w = np.ones([voc_size, weight_emb_size], dtype=floatX)
        init_w[0] = 0
        if init_type == 'idfs':
            init_w = options['idfs']
        return init_w

    def _get_node(self, node_name):
        if node_name in self.inputs_nodes:
            return self.inputs_nodes[node_name]
        elif node_name in self.nodes:
            return self.nodes[node_name]
        else:
            raise RuntimeError

    def _add_input(self):
        config = self.model_config['PHM']
        inputs_dict = self.inputs_nodes
        inputs_dict['input_story'] = Input(shape=(self.max_num_sentence, self.story_sentence_maxlen),
                                           name='input_story',
                                           dtype='int32')
        inputs_dict['input_question'] = Input(shape=(self.query_maxlen,), name='input_question', dtype='int32')
        inputs_dict['input_answer'] = Input(shape=(self.answer_size, self.answer_maxlen), name='input_answer',
                                            dtype='int32')
        inputs_dict['input_question_answer'] = Input(shape=(self.answer_size, self.qa_maxlen),
                                                     name='input_question_answer',
                                                     dtype='int32')
        inputs_dict['input_negation_questions'] = Input(shape=(self.answer_size, ),
                                                        name='input_negation_questions',
                                                        dtype=K.floatx())
        if config['use_slide_window_word']:
            inputs_dict['input_story_attentive'] = Input((self.story_maxlen,), name='input_story_attentive',
                                                         dtype='int32')
        if config['use_slide_window_reordered_word']:
            inputs_dict['input_reordered_story_attentive'] = Input((self.story_maxlen,),
                                                                   name='input_reordered_story_attentive',
                                                                   dtype='int32')
        if config['use_depend_score']:
            inputs_dict['input_dep'] = Input((self.max_num_sentence, self.answer_size), name='input_dep', dtype='int32')
            for ngram in config['ngram_inputs']:
                input_name = 'input_dep_%sgram' % ngram
                inputs_dict[input_name] = Input((self.max_num_sentence - ngram + 1, self.answer_size),
                                                name=input_name,
                                                dtype=K.floatx())

        # TODO: fix self.story_sentence_maxlen,
        if config['use_slide_window_inside_sentence']:
            inputs_dict['input_reordered_story'] = Input((self.max_num_sentence, self.story_sentence_maxlen),
                                                         name='input_reordered_story', dtype='int32')
            for ngram in config['ngram_inputs']:
                input_name = 'input_reordered_story_%sgram' % ngram
                max_len = self.data_options[input_name + "_shape"][2]
                inputs_dict[input_name] = Input((self.max_num_sentence - ngram + 1, max_len),
                                                name=input_name, dtype='int32')
        for ngram in config['ngram_inputs']:
            input_name = 'input_story_%sgram' % ngram
            max_len = self.data_options[input_name + "_shape"][2]
            inputs_dict[input_name] = Input(shape=(self.max_num_sentence - ngram + 1, max_len),
                                            name=input_name, dtype='int32')
        return inputs_dict

    def _add_embedding(self):
        # prepare embedding layers.
        nodes = self.nodes
        config = self.model_config['layers']['embedding']
        embedding_size = config['embedding_size']
        embedding_layer = MEmbedding(self.vocab_size, embedding_size,
                                     mask_zero=False,
                                     W_constraint=PadStayZero(),
                                     weights=self.W2V,
                                     trainable=False)

        have_input_question_answer = 'input_question_answer' in self.inputs_nodes
        have_input_stories_attentive = 'input_story_attentive' in self.inputs_nodes
        have_input_reordered_stories_attentive = 'input_reordered_story_attentive' in self.inputs_nodes
        have_input_reordered_stories = 'input_reordered_story' in self.inputs_nodes
        inputs = ['input_story', 'input_story', 'input_question', 'input_answer']
        outputs = ['story_word_embedding1', 'story_word_embedding2', 'question_word_embedding', 'answer_word_embedding']

        if have_input_question_answer:
            outputs.append('qa_word_embedding')
            inputs.append('input_question_answer')
        if have_input_stories_attentive:
            outputs.append('story_attentive_word_embedding')
            inputs.append('input_story_attentive')
        if have_input_reordered_stories_attentive:
            outputs.append('reordered_story_attentive_word_embedding')
            inputs.append('input_reordered_story_attentive')
        if have_input_reordered_stories:
            outputs.append('reordered_story_word_embedding')
            inputs.append('input_reordered_story')
            for ngram in self.model_config['PHM']['ngram_inputs']:
                inputs.append('input_reordered_story_%sgram' % ngram)
                outputs.append('reordered_story_word_embedding_%sgram' % ngram)

        # add ngram inputs
        for ngram in self.model_config['PHM']['ngram_inputs']:
            inputs.append('input_story_%sgram' % ngram)
            outputs.append('story_word_embedding1_%sgram' % ngram)
            if have_input_reordered_stories:
                inputs.append('input_reordered_story_%sgram' % ngram)
                outputs.append('reordered_story_word_embedding_%sgram' % ngram)

        for input, output in zip(inputs, outputs):
            nodes[output] = embedding_layer(self._get_node(input))

    def _add_sentence_encode(self, encode='sum'):
        assert encode in ['sum', 'weight-sum']
        nodes = self.nodes
        config = self.model_config['layers']['sentence_encode']
        trainable = config['trainable']

        weight_emb_size = 1
        trainable_embedding = config['trainable']
        init_W = [self._init_word_weight(init_type=config['init_type'])]
        w_story1 = MEmbedding(self.vocab_size, weight_emb_size, mask_zero=False,
                              W_constraint=PadStayZeroNonNeg(),
                              weights=init_W,
                              trainable=trainable_embedding)

        inputs = ['input_story', 'input_story', 'input_question', 'input_answer']
        outputs = ['__w_story1', '__w_story2', '__w_question', '__w_answer']
        for ngram in self.model_config['PHM']['ngram_inputs']:
            inputs.append('input_story_%sgram' % ngram)
            outputs.append('__w_story1_%sgram' % ngram)
        for input, output in zip(inputs, outputs):
            nodes[output] = w_story1(self._get_node(input))

        if encode == 'sum':
            nodes['story_encoding1'] = TimeDistributedMerge(axis=2, mode='sum')(nodes['story_word_embedding1'])
            nodes['answer_encoding'] = TimeDistributedMerge(axis=2, mode='sum')(nodes['answer_word_embedding'])
            nodes['question_encoding'] = TimeDistributedMerge(axis=1, mode='sum')(nodes['question_word_embedding'])
            for ngram in self.model_config['PHM']['ngram_inputs']:
                nodes['story_encoding1_%sgram' % ngram] = \
                    TimeDistributedMerge(axis=2, mode='sum')(nodes['story_word_embedding1_%sgram' % ngram])
        elif encode == 'weight-sum':
            weighting_layer = EmbeddingWeighting()
            nodes['__story_encoding1'] = weighting_layer([nodes['story_word_embedding1'], nodes['__w_story1']])
            nodes['__answer_encoding'] = weighting_layer([nodes['answer_word_embedding'], nodes['__w_answer']])
            nodes['__question_encoding'] = weighting_layer([nodes['question_word_embedding'], nodes['__w_question']])

            nodes['story_encoding1'] = TimeDistributedMerge(axis=2, mode='sum')(nodes['__story_encoding1'])
            nodes['answer_encoding'] = TimeDistributedMerge(axis=2, mode='sum')(nodes['__answer_encoding'])
            nodes['question_encoding'] = TimeDistributedMerge(axis=1, mode='sum')(nodes['__question_encoding'])

            for ngram in self.model_config['PHM']['ngram_inputs']:
                nodes['__story_encoding1_%sgram' % ngram] = weighting_layer(
                    [nodes['story_word_embedding1_%sgram' % ngram], nodes['__w_story1_%sgram' % ngram]])
                nodes['story_encoding1_%sgram' % ngram] = TimeDistributedMerge(axis=2, mode='sum')(
                    nodes['__story_encoding1_%sgram' % ngram])

        # add independent weight in wordbyword
        config = self.model_config['PHM']
        init_W = [self._init_word_weight()]
        trainable = config['trainable_qa_idf_weight'] and (
        config['wordbyword_merge_type'] == 'weighted_average' or config['use_qa_idf'])

        # q a weight for wordbyword matching.
        w_question_answer = MEmbedding(self.vocab_size, 1, mask_zero=False, W_constraint=PadStayZeroNonNeg(),
                                       weights=init_W,
                                       trainable=trainable)
        inputs = ['input_question', 'input_answer', 'input_question_answer']
        outputs = ['__w_question_wbw', '__w_answer_wbw', '__w_question_answer']
        for input, output in zip(inputs, outputs):
            nodes[output] = w_question_answer(self._get_node(input))

    def _add_model(self):
        nodes = self.nodes
        config = self.model_config['PHM']
        p = config['dropout_p']
        mlp_l2 = config['l2']
        D = config['mlp_output_dim']

        activation = lambda x: relu(x, alpha=config['leaky_alpha'])
        # SENTENCE LEVEL
        # answer plus question
        nodes['question_encoding_repeated'] = RepeatVector(self.answer_size)(nodes['question_encoding'])
        nodes['answer_plus_question'] = merge([nodes['answer_encoding'], nodes['question_encoding_repeated']],
                                              mode='sum')

        # story mlp and dropout
        ninputs, noutputs = ['story_encoding1'], ['story_encoding_mlp']
        for ngram in config['ngram_inputs']:
            ninputs.append('story_encoding1_%sgram' % ngram)
            noutputs.append('story_encoding_mlp_%sgram' % ngram)

        story_encoding_mlp = NTimeDistributed(Dense(D, init="identity", activation=activation,
                                                    W_regularizer=l2(mlp_l2),
                                                    trainable=config['trainable_story_encoding_mlp']))
        for input, output in zip(ninputs, noutputs):
            nodes[output] = story_encoding_mlp(self._get_node(input))
        qa_encoding_mlp = NTimeDistributed(Dense(D, init="identity", activation=activation,
                                                 W_regularizer=l2(mlp_l2),
                                                 trainable=config['trainable_answer_plus_question_mlp']))

        nodes['answer_plus_question_mlp'] = qa_encoding_mlp(nodes['answer_plus_question'])
        nodes['story_encoding_mlp_dropout'] = Dropout(p)(nodes['story_encoding_mlp'])
        nodes['answer_plus_question_mlp_dropout'] = Dropout(p)(nodes['answer_plus_question_mlp'])

        # norm
        unit_layer = UnitNormalization()
        nodes['story_encoding_mlp_dropout_norm'] = unit_layer(nodes['story_encoding_mlp_dropout'])
        nodes['answer_plus_question_norm'] = unit_layer(nodes['answer_plus_question_mlp_dropout'])
        # cosine
        nodes['story_dot_answer'] = merge([nodes['story_encoding_mlp_dropout_norm'],
                                           nodes['answer_plus_question_norm']],
                                          mode='dot', dot_axes=[2, 2])

        # WORD LEVEL
        # story mlps for word score and distance score
        trainable_word_mlp = self.model_config['PHM']['trainable_word_mlp']

        if trainable_word_mlp:
            story_word_dense = NTimeDistributed(
                Dense(D, init="identity", activation=activation, W_regularizer=l2(mlp_l2),
                      trainable=trainable_word_mlp), first_n=3)
            # q mlps for word and distance scores
            q_or_a_word_dense = NTimeDistributed(
                Dense(D, init="identity", activation=activation, W_regularizer=l2(mlp_l2),
                      trainable=trainable_word_mlp), first_n=3)
        else:
            linear_activation = Activation('linear')
            story_word_dense = linear_activation
            q_or_a_word_dense = linear_activation

        ninputs, noutputs = [], []
        tpls = [(True, 'story_word_embedding1', 'story_word_mlp'),
                ('use_slide_window_inside_sentence', 'reordered_story_word_embedding', 'reordered_story_word_mlp'),
                ('use_slide_window_word', 'story_attentive_word_embedding', 'story_attentive_word_embedding_mlp'),
                ('use_slide_window_reordered_word', 'reordered_story_attentive_word_embedding', 'reordered_story_attentive_word_embedding_mlp')
                ]
        for tpl in tpls:
            a, b, c = tpl
            if a is True or config[a]:
                ninputs.append(b)
                noutputs.append(c)
                if b in ['reordered_story_word_embedding', 'story_word_embedding1']:
                    for ngram in config['ngram_inputs']:
                        ninputs.append('%s_%sgram' % (b, ngram))
                        noutputs.append('%s_%sgram' % (c, ngram))

        for input, output in zip(ninputs, noutputs):
            nodes[output] = story_word_dense(self._get_node(input))
        inputs = ['question_word_embedding', 'answer_word_embedding', 'qa_word_embedding']
        outputs = ['question_word_mlp', 'answer_word_mlp', 'qa_word_mlp']
        for input, output in zip(inputs, outputs):
            nodes[output] = q_or_a_word_dense(self._get_node(input))

        # SIMILARITY MATRICES
        # first for word scores
        # cosine similarity matrix based on sentence and q
        nodes['sim_matrix_q'] = WordByWordMatrix(is_q=True)([nodes['story_word_mlp'], nodes['question_word_mlp']])

        # cosine similarity matrix based on sentence and a
        nodes['sim_matrix_a'] = WordByWordMatrix()([nodes['story_word_mlp'], nodes['answer_word_mlp']])

        # WORD-BY-WORD SCORES
        # q
        nodes['s_q_wbw_score'] = WordByWordScores(trainable=False, is_q=True, alpha=1., threshold=0.15,
                                                  wordbyword_merge_type=config['wordbyword_merge_type'],
                                                  )([nodes['sim_matrix_q'], nodes['__w_question_wbw']])
        # a
        nodes['s_a_wbw_score'] = WordByWordScores(trainable=False, alpha=1., threshold=0.15,
                                                  wordbyword_merge_type=config['wordbyword_merge_type'], )(
            [nodes['sim_matrix_a'], nodes['__w_answer_wbw']])
        # mean
        nodes['story_dot_answer_words'] = GeneralizedMean(mean_type=config['mean_type'],
                                                          trainable=config['trainable_story_dot_answer_words']) \
            ([nodes['s_q_wbw_score'], nodes['s_a_wbw_score']])

        # SLIDING WINDOW INSIDE SENTENCE
        if config['use_slide_window_inside_sentence']:
            # q+a mlp for word score
            # construct cosine similarity matrix based on sentence and qa, for word score
            _inputs = [nodes['reordered_story_word_mlp'], nodes['qa_word_mlp']]
            nodes['wordbyword_slide_sum_within_sentence'] = \
                WordByWordSlideSumInsideSentence(len(_inputs),
                                                 window_size=config['window_size_word_inside'],
                                                 alpha=config['alpha_slide_window_word_inside'],
                                                 use_gaussian_window=config['use_gaussian_window_word_inside'],
                                                 gaussian_std=config['gaussian_sd_word_inside'],
                                                 trainable=config['trainable_slide_window_word_inside'])(_inputs)

        # COMBINE LEVELS
        # sum word-based and sentence-based similarity scores
        inputs = ['story_dot_answer_words', 'story_dot_answer']
        if config['use_slide_window_sentence']:
            inputs.append('story_dot_answer_slide')
            nodes["story_dot_answer_slide"] = SlideSum(alpha=config['alpha_slide_window'],
                                                       use_gaussian_window=config['use_gaussian_window'],
                                                       trainable=config['trainable_slide_window'])(
                nodes['story_dot_answer'])

        if config['use_slide_window_inside_sentence']:
            inputs.append('wordbyword_slide_sum_within_sentence')

        if self.model_config['PHM']['use_depend_score']:
            # SENTENCE-QA DEPENDENCY LEVEL
            inputs.append('lcc_score_matrix')
            nodes['lcc_score_matrix'] = DependencyDistanceScore(config['alpha_depend_score'])(
                self._get_node('input_dep'))

        # sum scores from different component of the model on sentence level.
        # sentence level score merge
        layers_s_input = [nodes[x] for x in inputs]
        weights_s = [1.] * len(layers_s_input)
        nodes['word_plus_sent_sim'] = Combination(len(layers_s_input), input_dim=3, weights=weights_s,
                                                  combination_type=config['sentence_ensemble'],
                                                  trainable=config['trainable_sentence_ensemble'])(layers_s_input)

        # extract max over sentences
        nodes['story_dot_answer_max'] = TimeDistributedMerge(mode='max', axis=1)(nodes['word_plus_sent_sim'])

        # word sliding window
        word_sliding_window_output = ['story_dot_answer_max']
        if config['use_slide_window_word']:
            # q+a mlp for word score
            # construct cosine similarity matrix based on sentence and qa, for word score
            temp_inputs = [nodes['story_attentive_word_embedding_mlp'], nodes['qa_word_mlp']]
            if config['use_qa_idf']:
                temp_inputs.append(nodes['__w_question_answer'])
            nodes['wordbyword_slide_sum'] = WordByWordSlideSum(len(temp_inputs),
                                                               window_size=config['window_size_word'],
                                                               alpha=config['alpha_slide_window_word'],
                                                               use_gaussian_window=config['use_gaussian_window_word'],
                                                               gaussian_std=config['gaussian_sd_word'],
                                                               trainable=config['trainable_slide_window_word'])(
                temp_inputs)
            word_sliding_window_output.append('wordbyword_slide_sum')

        if config['use_slide_window_reordered_word']:
            # q+a mlp for word score
            # construct cosine similarity matrix based on sentence and qa, for word score
            temp_inputs = [nodes['reordered_story_attentive_word_embedding_mlp'], nodes['qa_word_mlp']]
            if config['use_qa_idf']:
                temp_inputs.append(nodes['__w_question_answer'])
            nodes['reordered_wordbyword_slide_sum'] = WordByWordSlideSum(len(temp_inputs),
                                                                         window_size=config[
                                                                             'window_size_reordered_word'],
                                                                         alpha=config[
                                                                             'alpha_slide_window_reordered_word'],
                                                                         use_gaussian_window=config[
                                                                             'use_gaussian_window_reordered_word'],
                                                                         gaussian_std=config[
                                                                             'gaussian_sd_reordered_word'],
                                                                         trainable=config[
                                                                             'trainable_slide_window_reordered_word'])(
                temp_inputs
                )
            word_sliding_window_output.append('reordered_wordbyword_slide_sum')

        # Extract top_n sentence for each answer
        if config['top_n_wordbyword']:
            layers_name = ['word_plus_sent_sim', 'story_word_embedding1', 'qa_word_embedding', '__w_question_answer']
            layers = [nodes[x] for x in layers_name]
            top_n_name = 'top_n_wordbyword'
            nodes[top_n_name] = TopNWordByWord(top_n=config['top_n'], nodes=nodes, use_sum=config['top_n_use_sum'],
                                               trainable=True)(layers)
            word_sliding_window_output.append(top_n_name)

        ngram_output = [self._add_ngram_network(ngram, story_encoding_mlp) for ngram in config['ngram_inputs']]

        # final score merge
        layers_input = [nodes[x] for x in word_sliding_window_output + ngram_output]
        weights = [1.] * len(layers_input)
        for i in range(len(ngram_output)):
            weights[-i - 1] = 1.

        """
        # also aggregate scores that were already aggregated on sentence level.
        sentence_level_weight = 0.1
        for layer_name in sentence_level_merge_layers:
            layer_max = layer_name + "_max"
            if layer_max not in nodes:
                add_node(TimeDistributedMergeEnhanced(mode='max'), layer_max, input=layer_name)
            layers_input.append(nodes[layer_max])
            weights.append(sentence_level_weight)"""

        nodes['story_dot_answer_combined_max'] = Combination(len(layers_input), weights=weights,
                                                             combination_type=config['answer_ensemble'],
                                                             trainable=config['trainable_answer_ensemble'])(
            layers_input)

        # apply not-switch
        input_mul = self._get_node('input_negation_questions')
        nodes['story_dot_answer_max_switch'] = merge([nodes['story_dot_answer_combined_max'], input_mul], mode='mul')

        activation_final = Activation('linear', name='y_hat') \
            if self.model_config['optimizer']['loss'] == 'ranking_loss' else Activation(
            'softmax', name='y_hat')
        prediction = activation_final(nodes['story_dot_answer_max_switch'])

        inputs = self.inputs_nodes.values()
        model = Model(input=inputs, output=prediction)
        optimizer = self._get_optimizer()
        model.compile(loss=self._get_loss_dict(), optimizer=optimizer, metrics={'y_hat': 'accuracy'})
        self.graph = model

    def _add_ngram_network(self, ngram, story_encoding_mlp_layer):
        nodes = self.nodes
        config = self.model_config['PHM']
        p = config['dropout_p']
        # activation = get_leaky_relu(config['leaky_alpha'])
        D = config['mlp_output_dim']

        story_encoding_mlp = 'story_encoding_mlp_%sgram' % ngram
        story_encoding_mlp_dropout = 'story_encoding_mlp_dropout_%sgram' % ngram
        story_encoding_mlp_dropout_norm = 'story_encoding_mlp_dropout_norm_%sgram' % ngram
        story_dot_answer = 'story_dot_answer_%sgram' % ngram
        story_word_mlp = 'story_word_mlp_%sgram' % ngram
        sim_matrix_q = 'sim_matrix_q_%sgram' % ngram
        sim_matrix_a = 'sim_matrix_a_%sgram' % ngram
        s_q_wbw_score = 's_q_wbw_score_%sgram' % ngram
        s_a_wbw_score = 's_a_wbw_score_%sgram' % ngram
        story_dot_answer_words = 'story_dot_answer_words_%sgram' % ngram
        word_plus_sent_sim = 'word_plus_sent_sim_%sgram' % ngram
        story_dot_answer_max = 'story_dot_answer_max_%sgram' % ngram
        reordered_story_word_mlp = 'reordered_story_word_mlp_%sgram' % ngram
        wordbyword_slide_sum_within_sentence = 'wordbyword_slide_sum_within_sentence_%sgram' % ngram
        lcc_score_matrix = 'lcc_score_matrix_%sgram' % ngram
        # SENTENCE LEVEL
        nodes[story_encoding_mlp_dropout] = Dropout(p)(nodes[story_encoding_mlp])

        # norm
        nodes[story_encoding_mlp_dropout_norm] = UnitNormalization()(nodes[story_encoding_mlp_dropout])
        # cosine
        nodes[story_dot_answer] = merge([nodes[story_encoding_mlp_dropout_norm],
                                         nodes['answer_plus_question_norm']],
                                        mode='dot', dot_axes=[2, 2])

        # WORD LEVEL
        # SIMILARITY MATRICES
        # first for word scores
        # cosine similarity matrix based on sentence and q
        nodes[sim_matrix_q] = WordByWordMatrix(is_q=True)([nodes[story_word_mlp], nodes['question_word_mlp']])
        # cosine similarity matrix based on sentence and a
        nodes[sim_matrix_a] = WordByWordMatrix()([nodes[story_word_mlp], nodes['answer_word_mlp']])

        # WORD-BY-WORD SCORES
        # q
        inputs = [nodes[sim_matrix_q], nodes['__w_question_wbw']]
        nodes[s_q_wbw_score] = WordByWordScores(is_q=True, alpha=1., threshold=0.15,
                                                wordbyword_merge_type=config['wordbyword_merge_type'],
                                                trainable=False,
                                                )(inputs)
        # a
        inputs = [nodes[sim_matrix_a], nodes['__w_answer_wbw']]
        nodes[s_a_wbw_score] = WordByWordScores(trainable=False, alpha=1., threshold=0.15,
                                                wordbyword_merge_type=config['wordbyword_merge_type'],
                                                )(inputs)

        # mean
        inputs = [nodes[s_q_wbw_score], nodes[s_a_wbw_score]]
        nodes[story_dot_answer_words] = GeneralizedMean(mean_type=config['mean_type'],
                                                        trainable=config['trainable_story_dot_answer_words'])(inputs)

        # SLIDING WINDOW INSIDE SENTENCE, not use this for now.
        if config['use_slide_window_inside_sentence']:
            # q+a mlp for word score
            # construct cosine similarity matrix based on sentence and qa, for word score
            inputs = [nodes[reordered_story_word_mlp], nodes['qa_word_mlp']]
            nodes[wordbyword_slide_sum_within_sentence] = \
                WordByWordSlideSumInsideSentence(len(inputs),
                                                 window_size=config['window_size_word_inside'],
                                                 alpha=config['alpha_slide_window_word_inside'],
                                                 use_gaussian_window=config['use_gaussian_window_word_inside'],
                                                 gaussian_std=config['gaussian_sd_word_inside'],
                                                 trainable=config['trainable_slide_window_word_inside'])(inputs)
        # COMBINE LEVELS
        # sum word-based and sentence-based similarity scores
        inputs = [story_dot_answer_words, story_dot_answer]
        # if config['use_slide_window_sentence']:
        #     add_node(SlideSum(alpha=config['alpha_slide_window'], use_gaussian_window=config['use_gaussian_window'],
        #                       trainable=config['trainable_slide_window']),
        #              story_dot_answer_slide, input=story_dot_answer)
        #     inputs.append(story_dot_answer_slide)

        if config['use_slide_window_inside_sentence']:
            inputs.append(wordbyword_slide_sum_within_sentence)

        if self.model_config['PHM']['use_depend_score']:
            # SENTENCE-QA DEPENDENCY LEVEL
            inputs.append(lcc_score_matrix)
            nodes[lcc_score_matrix] = DependencyDistanceScore(config['alpha_depend_score'])(
                self._get_node('input_dep_%sgram' % ngram))

        # sum scores from different component of the model on sentence level.
        # sentence level score merge
        layers_s_input = [nodes[x] for x in inputs]
        weights_s = [1.] * len(layers_s_input)
        nodes[word_plus_sent_sim] = Combination(len(layers_s_input), input_dim=3, weights=weights_s,
                                                combination_type=config['sentence_ensemble'],
                                                trainable=config['trainable_sentence_ensemble'])(layers_s_input)
        nodes[story_dot_answer_max] = TimeDistributedMerge(mode='max', axis=1)(nodes[word_plus_sent_sim])
        return story_dot_answer_max

    def _get_optimizer(self):
        optimizer_dict = self.model_config['optimizer']
        optimizer = optimizer_MAP[optimizer_dict['step_rule']]
        optimizer = optimizer(lr=optimizer_dict['learning_rate'], clipnorm=optimizer_dict['clipnorm'],
                              clipvalue=optimizer_dict['clipvalue'])
        return optimizer

    def _get_loss_dict(self):
        loss = self.model_config['optimizer']['loss']
        logger.info("using %s loss" % loss)
        if loss in ['ranking_loss']:
            return {'y_hat': ranking_loss(self.model_config['optimizer']['ranking_loss_margin'])}
        else:
            return {'y_hat': loss}

    def build(self):
        self._add_input()
        self._add_embedding()
        self._add_sentence_encode(encode=self.model_config['layers']['sentence_encode']['type'])
        self._add_model()
        self.print_parameters()
        return self.graph

    def print_parameters(self):
        from keras.engine.training import collect_trainable_weights
        logger.info("total number of parameters: %s" % self.graph.count_params())
        trainable_weights = collect_trainable_weights(self.graph)
        total = sum([K.count_params(p) for p in trainable_weights])
        logger.info("number of trainable parameters: %s" % total)


class ranking_loss(object):
    def __init__(self, margin, output_dim=4):
        self.margin = margin
        self.output_dim = output_dim
        self.__name__ = self.__class__.__name__

    def __call__(self, y_true, y_pred):
        return self.triplet_loss(y_true, y_pred)

    def triplet_loss(self, y_true, y_pred):
        y_pred = K.sigmoid(y_pred)

        p_plus = K.sum(y_true * y_pred, axis=1, keepdims=True)
        p_gaps = y_pred - p_plus + self.margin

        L = K.maximum(0, p_gaps)
        # return T.max(L, axis=1)
        return K.sum(L, axis=1)
