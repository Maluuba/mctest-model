# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import theano
from theano import tensor as T
import logging
from scipy import signal
import numpy as np

from keras.layers.core import Layer, Merge
from keras.layers.normalization import BatchNormalization
from keras import activations
from keras.layers.wrappers import TimeDistributed
from keras.engine.topology import to_list
from keras.engine import InputSpec
from keras.layers.wrappers import Wrapper
from keras import initializations, regularizers
from keras import backend as K

floatX = K.floatx()
logger = logging.getLogger(__name__)

floatX = theano.config.floatX

"""
put layers that are specific to option models here.
"""


def masked_softmax(x, m=None, axis=[-1]):
    '''
    Softmax with a mask that eliminates entries along the LAST dimension
    Inputs:
        x: ndim array
        m: ndim mask (optional)
    '''
    if m:
        x *= m
    x = K.clip(x, -5., 5.)
    e_x = K.exp(x - K.max(x, axis=axis, keepdims=True))
    if m:
        e_x = e_x * m
    softmax = e_x / (K.sum(e_x, axis=axis, keepdims=True) + 1e-6)
    return softmax


def masked_mean(inp, axis=None):
    # n_b x n_s x 4 x n_w_a: inp
    _mask = T.neq(inp, 0).astype(floatX)
    s_mask = T.sum(_mask, axis=axis, keepdims=True) + 0.00001  # to avoid nan error due to padded sentence.
    s = inp / s_mask
    s.name = 'mean'
    return T.sum(s, axis=axis)


def weighted_average(inp, weights, axis=None):
    # n_b x n_s x 4 x n_w_a: inp
    if axis == 2:  # for question
        weights = weights.flatten(ndim=2)
        weights /= T.sum(weights, axis=1, keepdims=True) + 0.000001
        return T.batched_tensordot(inp, weights, [[inp.ndim - 1], [1]])
    elif axis == 3:  # for answer inp: (None, 51, 4, 20), output: (None, 4, 20, 1)
        weights = weights.flatten(ndim=weights.ndim - 1)
        weights /= T.sum(weights, axis=weights.ndim - 1, keepdims=True) + 0.000001
        weights = weights.dimshuffle(0, 'x', 1, 2)
        return T.sum(inp * weights, axis=3)
    elif axis == 4:  # for inner sliding window
        weights = weights.flatten(ndim=weights.ndim - 1)
        weights /= T.sum(weights, axis=weights.ndim - 1, keepdims=True) + 0.000001
        weights = weights.dimshuffle(0, 'x', 'x', 1, 2)
        return T.sum(inp * weights, axis=4)
    else:
        raise RuntimeError


class WeightedSum(Layer):
    '''
    Weighted sum
    '''

    def __init__(self, alpha=1.0, beta=1.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(alpha)
        self.beta = K.variable(beta)

    def build(self, input_shape):
        self.trainable_weights = [self.alpha, self.beta]

    def get_output_shape_for(self, input_shape):
        return input_shape[0]

    def call(self, x, mask=None):
        return x[0] * self.alpha + x[1] * self.beta


class MaskPassThrough(object):
    def __init__(self, *args, **kwargs):
        self.supports_masking = True
        super(MaskPassThrough, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return input_mask


class TimeDistributedMerge(Layer):
    def __init__(self, mode='max', axis=1, **kwargs):
        assert mode in ['max', 'mean', 'sum', 'concat']
        assert axis in set([1, 2])
        self.mode = mode
        self.axis = axis
        self.support_masking = True
        super(TimeDistributedMerge, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.mode in ['max', 'mean', 'sum']:
            shape_list = list(input_shape)
            del shape_list[self.axis]
            return tuple(shape_list)
        elif self.mode in ['concat']:
            assert len(input_shape) >= 2
            stacked_dim = 0
            for inp_shape in input_shape[1:]:
                # Special treatment for 2D matrices
                if len(inp_shape) == 2:
                    stacked_dim += 1 if self.axis == 1 else inp_shape[-1]
                else:
                    stacked_dim += inp_shape[self.axis]
            return_shape = list(input_shape[0])
            return_shape[self.axis] += stacked_dim
            return tuple(return_shape)
        else:
            raise NotImplemented

    def call(self, x, mask=None):
        if self.mode == 'max':
            return K.max(x, axis=self.axis)
        elif self.mode == 'mean':
            return K.mean(x, axis=self.axis)
        elif self.mode == 'sum':
            return K.sum(x, axis=self.axis)
        elif self.mode == 'concat':
            assert len(x) >= 2
            assert x[0].ndim == 3

            def _transform(target):
                # Expand first dimension in any case
                target = K.expand_dims(target, dim=1)
                if self.axis == 2:
                    # Repeat target along the time dimension
                    target = K.repeat_elements(
                        target, x[0].shape[1], axis=1)
                return target

            targets = map(lambda t: _transform(t) if t.ndim == 2 else t, x[1:])
            return K.concatenate([x[0]] + targets, axis=self.axis)
        else:
            raise NotImplemented

    def compute_mask(self, input, input_mask=None):
        return None


class NTimeDistributed(Wrapper):
    def __init__(self, layer, first_n=2, **kwargs):
        self.supports_masking = True
        self.first_n = first_n
        super(NTimeDistributed, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis.')
        child_input_shape = (input_shape[0],) + input_shape[self.first_n:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(NTimeDistributed, self).build()

    def get_output_shape_for(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[self.first_n:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        timesteps = tuple(input_shape[1:self.first_n])
        return (child_output_shape[0],) + timesteps + child_output_shape[1:]

    def call(self, X, mask=None):
        input_shape = self.input_spec[0].shape
        # no batch size specified, therefore the layer will be able
        # to process batches of any size
        # we can go with reshape-based implementation for performance
        tensor_input_shape = K.shape(X)
        input_length = tuple(tensor_input_shape[:self.first_n])
        X = K.reshape(X, (-1,) + tuple(tensor_input_shape[self.first_n:]))  # nb_samples * ... *timesteps, ...)
        y = self.layer.call(X)  # (nb_samples * timesteps, ...)
        # (nb_samples, timesteps, ...)
        output_shape = self.get_output_shape_for(input_shape)
        y = K.reshape(y, input_length + output_shape[self.first_n:])
        return y

    def __call__(self, x, mask=None):
        '''
        when it is used as a shared layer and the input_shape changes,
        it will get a failure when assert_input_compatibility for the other inputs
        if the second dim is different the first one.
        The second dim shouldn't affect anything.
        '''
        if not self.built:
            # raise exceptions in case the input is not compatible
            # with the input_spec specified in the layer constructor
            self.assert_input_compatibility(x)

            # collect input shapes to build layer
            input_shapes = []
            for x_elem in to_list(x):
                if hasattr(x_elem, '_keras_shape'):
                    input_shapes.append(x_elem._keras_shape)
                elif hasattr(K, 'int_shape'):
                    input_shapes.append(K.int_shape(x_elem))
                else:
                    raise Exception('You tried to call layer "' + self.name +
                                    '". This layer has no information'
                                    ' about its expected input shape, '
                                    'and thus cannot be built. '
                                    'You can build it manually via: '
                                    '`layer.build(batch_input_shape)`')
            if len(input_shapes) == 1:
                self.build(input_shapes[0])
            else:
                self.build(input_shapes)
            self.built = True

        input_added = False
        input_tensors = to_list(x)

        inbound_layers = []
        node_indices = []
        tensor_indices = []
        for input_tensor in input_tensors:
            if hasattr(input_tensor, '_keras_history') and input_tensor._keras_history:
                # this is a Keras tensor
                previous_layer, node_index, tensor_index = input_tensor._keras_history
                inbound_layers.append(previous_layer)
                node_indices.append(node_index)
                tensor_indices.append(tensor_index)
            else:
                inbound_layers = None
                break
        if inbound_layers:
            # this will call layer.build() if necessary
            self.add_inbound_node(inbound_layers, node_indices, tensor_indices)
            input_added = True

        # get the output tensor to be returned
        if input_added:
            # output was already computed when calling self.add_inbound_node
            outputs = self.inbound_nodes[-1].output_tensors
            # if single output tensor: return it,
            # else return a list (at least 2 elements)
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs
        else:
            # this case appears if the input was not a Keras tensor
            return self.call(x, mask)


class EmbeddingWeighting(Layer):
    def call(self, x, mask=None):
        lay0 = x[0]
        lay1 = x[1]
        lay1 = T.addbroadcast(lay1, lay1.ndim - 1)
        return lay0 * lay1

    def get_output_shape_for(self, input_shape):
        return input_shape[0]


class UnitNormalization(Layer):
    input_dims = 3

    def __init__(self, smooth_factor=1e-6, **kwargs):
        super(UnitNormalization, self).__init__(**kwargs)
        self.smooth_factor = smooth_factor

    def call(self, x, mask=None):
        mag_X = K.sqrt(K.sum(x ** 2, axis=2, keepdims=True) + self.smooth_factor)
        return x / mag_X


class WordByWordMatrix(Layer):
    def __init__(self, is_q=False, **kwargs):
        ''' Compute word-by-word cosine similarities
            Return as matrix
        '''
        super(WordByWordMatrix, self).__init__(**kwargs)
        # if len(layers) not in [2, 3]:
        #     raise Exception("Please specify 3 input layers (or containers), not %s, to merge. " % len(layers))
        self.is_q = is_q

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 2
        if self.is_q:
            return (
                input_shape[0][0],  # n_b ~ None
                input_shape[0][1],  # n_s
                input_shape[0][2],  # n_w_s
                input_shape[1][1],  # n_w_qa
            )
        else:
            return (
                input_shape[0][0],  # n_b ~ None
                input_shape[0][1],  # n_s
                input_shape[1][1],  # 4
                input_shape[0][2],  # n_w_s
                input_shape[1][2],  # n_w_qa
            )

    def call(self, x, mask=None):
        ax = 1 if self.is_q else 2

        def _step(v1, v2):
            cosine_score = T.tensordot(v1 / T.sqrt(T.sum(T.sqr(v1), axis=2, keepdims=True) + 1e-6),
                                       (v2) / T.sqrt(T.sum(T.sqr(v2), axis=ax, keepdims=True) + 1e-6),
                                       [[2], [ax]])
            return cosine_score

        l_s = x[0]  # n_b x n_s x n_w_s x D
        l_a = x[1]  # n_b x 4 x n_w_qa x D
        # w_qa = self.layers[2].get_output(train)  # n_b x 4 x n_w_qa x 1
        # w_qa = T.addbroadcast(w_qa, len(self.layers[2].output_shape) - 1)

        # get cosine similarity for ALL word pairs
        output, _ = theano.scan(_step, sequences=[l_s, l_a], outputs_info=None)
        if not self.is_q:
            output = output.dimshuffle(0, 1, 3, 2, 4)  # n_b x n_s x 4 x n_w_s x n_w_qa
        return output


class WordByWordScores(Layer):
    '''
        Aggregate word-by-word scores given cosine similarity matrix
    '''

    def __init__(self, wordbyword_merge_type=False, alpha=1.0, is_q=False, threshold=0., **kwargs):
        super(WordByWordScores, self).__init__(**kwargs)
        self.alpha = K.variable(alpha)
        self.wordbyword_merge_type = wordbyword_merge_type
        self.is_q = is_q
        self.threshold = threshold
        self.output_cache = {}

    def build(self, input_shape):
        self.trainable_weights = [self.alpha]

    def get_output_shape_for(self, input_shape):
        if self.is_q:
            return input_shape[0][0], input_shape[0][1]
        else:
            return input_shape[0]

    def call(self, x, mask=None):
        X = x[0]  # shape is n_b x n_s x 4 x n_w_s x n_w_a

        ax = 2 if self.is_q else 3
        # reduce over n_w_s
        output = T.max(X, axis=ax)
        output = T.switch(T.gt(output, self.threshold), output, 0)
        # reduce over n_w_a
        if self.wordbyword_merge_type == 'max':
            output = T.max(output, axis=ax)  # get max max_sim for each a  # n_b x n_s x 4
        elif self.wordbyword_merge_type == 'mask_average':
            output = masked_mean(output, axis=ax)  # get average max_sim for each a
        elif self.wordbyword_merge_type == 'weighted_average':
            weight_layer = x[1]
            output = weighted_average(output, weight_layer, axis=ax)
        output = output * self.alpha
        return output


class GeneralizedMean(Layer):
    def __init__(self, alpha=1.25, beta=1.75, gama=0.0, mean_type='arithmetic', **kwargs):
        ''' Compute generalized mean of two scores
        '''
        super(GeneralizedMean, self).__init__(**kwargs)
        # if len(layers) < 2:
        #     raise Exception("Please specify two or more input layers (or containers) to merge")
        assert mean_type in ['harmonic', 'arithmetic', 'geometric', 'bilinear']
        self.alpha = K.variable(alpha)
        self.beta = K.variable(beta)
        self.gama = K.variable(gama)
        self.mean_type = mean_type
        # self.layers = layers
        # self.build()  # this has to be called here since GeneralizedMean doesn't has a previous.

    def build(self, input_shape):
        self.trainable_weights = [self.alpha, self.beta, self.gama]

    def get_output_shape_for(self, input_shape):
        return (
            input_shape[1][0],  # n_b ~ None
            input_shape[1][1],  # n_s
            input_shape[1][2]  # 4
        )

    def call(self, x, mask=None):
        l_q = x[0]  # n_b x n_s
        l_a = x[1]  # n_b x n_s x 4
        # add broadcast dimension to end of l_q
        l_q = l_q.dimshuffle(0, 1, 'x')

        if self.mean_type == 'harmonic':
            # compute harmonic mean of two scores
            output = 2. * l_q * l_a / (l_q + l_a + 0.00001) * self.beta
        elif self.mean_type == 'geometric':
            # compute geometric mean of two scores
            output = T.sqrt(l_q * l_a + 0.00001) * self.beta
        elif self.mean_type == 'bilinear':
            output = l_q * l_a * self.alpha + self.beta * l_a + self.gama * l_q
        else:
            # compute arithmetic mean
            output = (l_q + l_a) / 2.

        return output + 0 * (self.alpha + self.beta + self.gama)


class WordByWordSlideSumInsideSentence(Layer):
    def __init__(self, window_size=60, alpha=1.6, use_gaussian_window=True, gaussian_std=10, **kwargs):
        ''' Compute word-by-word cosine similarities within sentence
            Return as matrix
        '''
        super(WordByWordSlideSumInsideSentence, self).__init__(**kwargs)
        self.use_qa_idf = False
        if len(self.layers) == 3:
            self.use_qa_idf = True
        self.window_size = int(window_size)
        self.alpha = K.variable(alpha)
        self.w_gaussian = None
        self.use_gaussian_window = use_gaussian_window
        self.std = gaussian_std

    def build(self):
        self.trainable_weights = [self.alpha]
        if self.use_gaussian_window:
            window = signal.gaussian(self.window_size, std=self.std)
        else:
            window = np.ones(self.window_size, dtype=floatX)
        self.w_gaussian = K.variable(window)
        self.trainable_weights.append(self.w_gaussian)

    def get_output_shape_for(self, input_shape):
        return (
            input_shape[0][0],  # n_b
            input_shape[0][1],  # n_s
            input_shape[1][1],  # 4
        )

    def call(self, x, mask=None):
        def _step(v1, v2):
            cosine_score = T.tensordot(v1 / T.sqrt(T.sum(T.sqr(v1), axis=2, keepdims=True) + 1e-6),
                                       (v2) / T.sqrt(T.sum(T.sqr(v2), axis=2, keepdims=True) + 1e-6),
                                       [[2], [2]])
            return cosine_score

        l_s = x[0]  # n_b x n_s x n_w_s x D
        l_a = x[1]  # n_b x 4 x n_w_qa x D
        # get cosine similarity for ALL word pairs
        output, _ = theano.scan(_step, sequences=[l_s, l_a], outputs_info=None)  # n_b x n_s x n_w_s x 4 x n_w_qa
        # return T.max(T.max(output, axis=4), axis=2)
        output = output.dimshuffle(2, 1, 0, 3, 4)  # n_w_s x n_s x n_b x 4 x n_w_qa

        def slide_max(i, X):
            size = self.window_size
            M = X[i:i + size]
            W = self.w_gaussian
            return T.max((W * M.T).T, axis=0), theano.scan_module.until(i >= X.shape[0] - size + 1)

        output, _ = theano.scan(slide_max,
                                sequences=[
                                    T.arange(0, stop=(output.shape[0] - self.window_size + 1), step=3, dtype='int32')],
                                non_sequences=output)
        if self.use_qa_idf:
            average = weighted_average(output.dimshuffle(2, 1, 0, 3, 4), x[2], axis=4)
        else:
            average = masked_mean(output.dimshuffle(2, 1, 0, 3, 4), axis=4)
        return T.max(average, axis=2) * self.alpha
        # return T.max(masked_mean(output.dimshuffle(2, 1, 0, 3, 4), axis=4), axis=2) * self.alpha


class SlideSum(Layer):
    '''
        Dimensions of input are assumed to be (nb_samples, dim, 1).
        Return tensor of shape (nb_samples, dim - window_size + 1, n).
    '''

    def __init__(self, window_size=3, alpha=0.7, use_gaussian_window=False, **kwargs):
        super(SlideSum, self).__init__(**kwargs)
        self.window_size = window_size
        self.alpha = K.variable(alpha)
        self.w_gaussian = None
        self.use_gaussian_window = use_gaussian_window

    def build(self, input_shape):
        self.trainable_weights = [self.alpha]
        if self.use_gaussian_window:
            self.std = 2
            window = signal.gaussian(self.window_size, std=self.std)
            self.w_gaussian = K.variable(window)

    def get_output_shape_for(self, input_shape):
        input_shape_list = list(input_shape)
        # input_shape_list[1] -= self.window_size - 1
        return tuple(input_shape_list)

    def call(self, x, mask=None):
        def slide_sum(i, X):
            size = self.window_size
            if self.use_gaussian_window:
                # M = theano.printing.Print('X')(X[i:i+size])
                M = X[i:i + size]
                W = self.w_gaussian
                return T.dot(W, M)
                # , theano.scan_module.until(i >= X.shape[0] - size)
            else:
                return T.sum(X[i:i + size], axis=0)
                # , theano.scan_module.until(i >= X.shape[0] - size)

        def inner_scan(X):
            Y, _ = theano.scan(slide_sum, sequences=[T.arange(X.shape[0] - self.window_size + 1, dtype='int32')],
                               non_sequences=X)
            return Y

        X = x
        Y, _ = theano.scan(inner_scan, sequences=X)

        pad_len = (self.window_size - 1) / 2
        Y = T.concatenate([Y[:, :pad_len, :], Y, Y[:, Y.shape[1] - pad_len:, :]], axis=1)
        return Y * self.alpha


class DependencyDistanceScore(Layer):
    '''
        Aggregate pre-calculated dependency distance matrix, and make the coefficient trainable
    '''

    def __init__(self, alpha=1.0, **kwargs):
        super(DependencyDistanceScore, self).__init__(**kwargs)
        self.alpha = K.variable(alpha)

    def build(self, input_shape):
        self.trainable_weights = [self.alpha]

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]

    def call(self, x, mask=None):
        return x * self.alpha


class Combination(Layer):
    '''
    combine scores from different component.
    '''

    def __init__(self, layers_len, input_dim=2, weights=[], combination_type='sum', **kwargs):
        """
        :param layers:
        :param input_dim: if 2, means final 'answer level' combination, 3, sentence level scores combination.
        :param weights:
        :param combination_type:
        :param kwargs:
        :return:
        """
        super(Combination, self).__init__(**kwargs)
        if layers_len < 2:
            raise Exception("Please specify two or more input layers (or containers) to merge")
        assert layers_len == len(weights) or len(weights) == 0
        self.layers_len = layers_len
        assert combination_type in ['sum', 'mlp', 'bilinear']
        self.weights = weights if len(weights) == layers_len else [1.] * layers_len
        self.combination_type = combination_type
        self.input_dim = input_dim

    def build(self, input_shape):
        if self.combination_type == 'mlp':
            # self.mlp = TimeDistributedDense(1, init='one',
            #                                 weights=[np.expand_dims(self.weights, axis=1), np.zeros(1)])
            # self.mlp.set_input_shape([None, 4, len(self.layers)])

            self.W = K.variable(np.expand_dims(self.weights, axis=1), dtype=floatX)
            self.trainable_weights.append(self.W)

        elif self.combination_type == 'bilinear':
            self.BW = K.variable(np.diag(self.weights))
            self.trainable_weights.append(self.BW)

    def get_output_shape_for(self, input_shape):
        return input_shape[0]

    def call(self, x, mask=None):
        num_layers = self.layers_len
        inputs = x
        if self.combination_type == 'sum':
            inputs = [w * x for x, w in zip(inputs, self.weights)]
        X = T.concatenate(inputs, axis=inputs[0].ndim - 1)
        if self.input_dim == 2:
            X = X.reshape([X.shape[0], num_layers, X.shape[1] // num_layers])
            X = X.dimshuffle(0, 2, 1)  # nb_b * nb_a * nb_num_layers (num of scores to combine)
        elif self.input_dim == 3:
            X = X.reshape([X.shape[0], X.shape[1], num_layers, X.shape[2] // num_layers])
            X = X.dimshuffle(0, 1, 3, 2)  # nb_b * nb_a * nb_num_layers (num of scores to combine)

        if self.combination_type == 'sum':
            output = T.sum(X, axis=X.ndim - 1)
        elif self.combination_type == 'mlp':
            # self.mlp.input = X
            # output = self.mlp.get_output(train)
            output = T.dot(X, self.W)
            output = output.flatten(output.ndim - 1)
        elif self.combination_type == 'bilinear':
            Y = T.dot(X, self.BW)
            output = T.sum(X * Y, axis=Y.ndim - 1)
        return output

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "combination_type": self.combination_type,
                  "layers": [l.get_config() for l in self.layers]}
        base_config = super(Combination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WordByWordSlideSum(Layer):
    def __init__(self, layers_len, window_size=60, alpha=1.6, use_gaussian_window=True, gaussian_std=10, **kwargs):
        ''' Compute word-by-word cosine similarities
            Return as matrix
        '''
        super(WordByWordSlideSum, self).__init__(**kwargs)
        self.use_qa_idf = False
        if layers_len == 3:
            self.use_qa_idf = True
        self.window_size = int(window_size)
        self.alpha = K.variable(alpha)
        self.w_gaussian = None
        self.use_gaussian_window = use_gaussian_window
        self.std = gaussian_std

    def build(self, input_shape):
        self.trainable_weights = [self.alpha]
        if self.use_gaussian_window:
            window = signal.gaussian(self.window_size, std=self.std)
        else:
            window = np.ones(self.window_size, dtype=floatX)
        self.w_gaussian = K.variable(window)
        self.trainable_weights.append(self.w_gaussian)

    def get_output_shape_for(self, input_shape):
        return (
            input_shape[0][0],  # n_b ~ None
            input_shape[1][1],  # 4
            # self.layers[1].output_shape[2]   # n_w_qa
        )

    def call(self, x, mask=None):
        def _step(v1, v2):
            cosine_score = T.tensordot(v1 / T.sqrt(T.sum(T.sqr(v1), axis=1, keepdims=True) + 1e-6),
                                       (v2) / T.sqrt(T.sum(T.sqr(v2), axis=2, keepdims=True) + 1e-6),
                                       [[1], [2]])
            return cosine_score

        l_s = x[0]  # n_b x n_w_st x D
        l_a = x[1]  # n_b x 4 x n_w_qa x D
        # get cosine similarity for ALL word pairs
        output, _ = theano.scan(_step, sequences=[l_s, l_a], outputs_info=None)  # n_b x n_w_st x 4 x n_w_qa
        output = output.dimshuffle(1, 0, 2, 3)

        def slide_max(i, X):
            size = self.window_size
            M = X[i:i + size]
            W = self.w_gaussian
            return T.max((W * M.T).T, axis=0), theano.scan_module.until(i >= X.shape[0] - size + 1)

        output, _ = theano.scan(slide_max,
                                sequences=[
                                    T.arange(0, stop=(output.shape[0] - self.window_size + 1), step=5, dtype='int32')],
                                non_sequences=output)
        if self.use_qa_idf:
            average = weighted_average(output.dimshuffle(1, 0, 2, 3), x[2], axis=3)
        else:
            average = masked_mean(output.dimshuffle(1, 0, 2, 3), axis=3)
        return T.max(average, axis=1) * self.alpha

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "layers": [l.get_config() for l in self.layers]}
        base_config = super(WordByWordSlideSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WordByWordSlideSumInsideSentence(Layer):
    def __init__(self, layers_len, window_size=60, alpha=1.6, use_gaussian_window=True, gaussian_std=10, **kwargs):
        ''' Compute word-by-word cosine similarities within sentence
            Return as matrix
        '''
        super(WordByWordSlideSumInsideSentence, self).__init__(**kwargs)
        self.use_qa_idf = False
        if layers_len == 3:
            self.use_qa_idf = True
        self.window_size = int(window_size)
        self.alpha = K.variable(alpha)
        self.w_gaussian = None
        self.use_gaussian_window = use_gaussian_window
        self.std = gaussian_std

    def build(self, input_shape):
        self.trainable_weights = [self.alpha]
        if self.use_gaussian_window:
            window = signal.gaussian(self.window_size, std=self.std)
        else:
            window = np.ones(self.window_size, dtype=floatX)
        self.w_gaussian = K.variable(window)
        self.trainable_weights.append(self.w_gaussian)

    def get_output_shape_for(self, input_shape):
        return (
            input_shape[0][0],  # n_b
            input_shape[0][1],  # n_s
            input_shape[1][1],  # 4
        )

    def call(self, x, mask=None):
        def _step(v1, v2):
            cosine_score = T.tensordot(v1 / T.sqrt(T.sum(T.sqr(v1), axis=2, keepdims=True) + 1e-6),
                                       (v2) / T.sqrt(T.sum(T.sqr(v2), axis=2, keepdims=True) + 1e-6),
                                       [[2], [2]])
            return cosine_score

        l_s = x[0]  # n_b x n_s x n_w_s x D
        l_a = x[1]  # n_b x 4 x n_w_qa x D
        # get cosine similarity for ALL word pairs
        output, _ = theano.scan(_step, sequences=[l_s, l_a], outputs_info=None)  # n_b x n_s x n_w_s x 4 x n_w_qa
        # return T.max(T.max(output, axis=4), axis=2)
        output = output.dimshuffle(2, 1, 0, 3, 4)  # n_w_s x n_s x n_b x 4 x n_w_qa

        def slide_max(i, X):
            size = self.window_size
            M = X[i:i + size]
            W = self.w_gaussian
            return T.max((W * M.T).T, axis=0), theano.scan_module.until(i >= X.shape[0] - size + 1)

        output, _ = theano.scan(slide_max,
                                sequences=[
                                    T.arange(0, stop=(output.shape[0] - self.window_size + 1), step=3, dtype='int32')],
                                non_sequences=output)
        if self.use_qa_idf:
            average = weighted_average(output.dimshuffle(2, 1, 0, 3, 4), x[2], axis=4)
        else:
            average = masked_mean(output.dimshuffle(2, 1, 0, 3, 4), axis=4)
        return T.max(average, axis=2) * self.alpha
        # return T.max(masked_mean(output.dimshuffle(2, 1, 0, 3, 4), axis=4), axis=2) * self.alpha

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "layers": [l.get_config() for l in self.layers]}
        base_config = super(WordByWordSlideSumInsideSentence, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TopNWordByWord(Layer):
    def __init__(self, top_n, alpha=1.5, beta=1., nodes=None, use_sum=True, **kwargs):
        ''' Compute word-by-word cosine similarities
            Return as matrix
        '''
        super(TopNWordByWord, self).__init__(**kwargs)
        self.alpha = K.variable(alpha)
        self.beta = K.variable(beta)
        self.top_n = int(top_n)
        self.top_n_s_ids = None
        self.nodes = nodes
        self.use_sum = use_sum

    def build(self, input_shape):
        self.trainable_weights = [self.alpha, self.beta]

    def get_output_shape_for(self, input_shape):
        return (
            input_shape[0][0],  # n_b ~ None
            input_shape[0][2]  # 4
        )

    def call(self, x, mask=None):
        # layers_name = ['word_plus_sent_sim', 'story_word_embedding1', 'qa_word_embedding', '__w_question_answer']
        top_n = self.top_n
        sentence_scores = x[0]  # n_b * n_s * 4
        story_word_embedding = x[1]  # n_b * n_s * n_w * n_e
        qa_embedding = x[2]  # n_b * 4 * n_w * n_e
        qa_weights = x[3]  # n_b * 4 * n_w * 1
        top_n_s = T.argsort(sentence_scores, axis=1)[:, -top_n:]  # n_b * top_n * 4
        self.top_n_s_ids = top_n_s

        def _step(emb, idx):
            shape0, shape1 = idx.shape
            ret_emb = emb[idx.flatten()]
            return ret_emb.reshape((shape0, shape1, emb.shape[1], emb.shape[2]))

        # n_b x top_n * 4 * n_w * n_e, top_n sentence for each choice.
        output, _ = theano.scan(_step, sequences=[story_word_embedding, top_n_s], outputs_info=None)
        output = output.dimshuffle(0, 2, 1, 3, 4)  # n_b * 4  * top_n * n_w * n_e
        shapes = output.shape
        # n_b * 4 * top_n-n_w * n_ex
        top_n_s_emb = output.reshape([shapes[0], shapes[1], shapes[2] * shapes[3], shapes[4]])

        if self.use_sum:
            w_story1 = self.nodes['__w_story1']  # n_b * n_s * n_w * 1
            qa_encoding = self.nodes['answer_plus_question']  # n_b * 4 * n_e
            top_w_story1, _ = theano.scan(_step, sequences=[w_story1, top_n_s])  # n_b * top_n * 4 * n_w * 1
            top_w_story1 = top_w_story1.dimshuffle(0, 2, 1, 3, 4)  # n_b * 4  * top_n * n_w * 1
            # top_w_story1 = top_w_story1.dimshuffle(0, 'x', 1, 2, 3)  # n_b * 1 * n_top * n_w * 1
            # shapes = theano.printing.Print('dim:')(top_w_story1.shape)
            shapes = top_w_story1.shape
            # n_b * 4 * top_n-n_w * 1
            top_w_story1 = top_w_story1.reshape([shapes[0], shapes[1], shapes[2] * shapes[3], shapes[4]])
            top_w_story1 = T.addbroadcast(top_w_story1, top_w_story1.ndim - 1)

            top_n_s_encoding = T.sum(top_n_s_emb * top_w_story1, axis=2)  # n_b * 4 * n_e

            qa_encoding = qa_encoding / T.sqrt(T.sum(T.sqr(qa_encoding), axis=2, keepdims=True) + 1e-6)
            top_n_s_encoding = qa_encoding / T.sqrt(T.sum(T.sqr(top_n_s_encoding), axis=2, keepdims=True) + 1e-6)
            sum_cosine = T.sum(qa_encoding * top_n_s_encoding, axis=2) * self.beta
        else:
            sum_cosine = 0 * self.beta

        shapes = qa_embedding.shape
        qa_embedding = qa_embedding.reshape([shapes[0] * shapes[1], shapes[2], shapes[3]])
        shapes = top_n_s_emb.shape
        top_n_s_emb = top_n_s_emb.reshape([shapes[0] * shapes[1], shapes[2], shapes[3]])

        def _step_cosine(v1, v2):
            cosine_score = T.tensordot(v1 / T.sqrt(T.sum(T.sqr(v1), axis=1, keepdims=True) + 1e-6),
                                       v2 / T.sqrt(T.sum(T.sqr(v2), axis=1, keepdims=True) + 1e-6),
                                       [[1], [1]])
            return cosine_score

        # 4n_b * n_w * top_n-n_w
        cosine_matrix, _ = theano.scan(_step_cosine, sequences=[qa_embedding, top_n_s_emb], outputs_info=None)
        cosine_matrix_max = T.max(cosine_matrix, axis=2)
        c_shapes = cosine_matrix_max.shape
        cosine_matrix_max = cosine_matrix_max.reshape([shapes[0], shapes[1], c_shapes[1]])

        weights = qa_weights
        weights = weights.flatten(ndim=weights.ndim - 1)
        weights /= T.sum(weights, axis=weights.ndim - 1, keepdims=True) + 0.000001
        return T.sum(cosine_matrix_max * weights, axis=2) * self.alpha + sum_cosine

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "layers": [l.get_config() for l in self.layers]}
        base_config = super(TopNWordByWord, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
