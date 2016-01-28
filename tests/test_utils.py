# -*- coding: UTF-8 -*-
import numpy as np

import theano

import dl


def test_to_float_X():
    x = np.asarray(np.random.normal(size=(10,5)))
    assert dl.utils.to_float_X(x).dtype == theano.config.floatX


def test_shared_variable():
    x = np.asarray(np.random.normal(size=(10,5)))
    assert isinstance(dl.utils.shared_variable(x), theano.compile.SharedVariable)
