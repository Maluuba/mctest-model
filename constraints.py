# encoding: utf-8
from __future__ import absolute_import
from keras import backend as K
from keras.constraints import Constraint
from theano import tensor as T


class PadStayZero(Constraint):
    def __call__(self, p):
        T.set_subtensor(p[0:1], 0, inplace=True)
        return p


class PadStayZeroNonNeg(Constraint):
    def __call__(self, p):
        T.set_subtensor(p[0:1], 0, inplace=True)
        p *= K.cast(p >= 0., K.floatx())
        return p
