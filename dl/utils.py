# -*- coding: UTF-8 -*-


import numpy as np

import theano


_float_X = theano.config.floatX
_EPSILON = 1e-8


def to_float_X(arr):
    return np.asarray(arr, dtype=_float_X)


def shared_variable(value, dtype=_float_X, name=None, borrow=True):
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, borrow=True)

