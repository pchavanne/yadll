# -*- coding: UTF-8 -*-


import numpy as np

import theano


floatX = theano.config.floatX
EPSILON = 1e-8


def to_float_X(arr):
    return np.asarray(arr, dtype=floatX)


def shared_variable(value, dtype=floatX, name=None, borrow=True):
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, borrow=True)

