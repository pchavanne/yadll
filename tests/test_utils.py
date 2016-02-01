# -*- coding: UTF-8 -*-
import time
import numpy as np

import theano

import dl


def test_to_float_X():
    x = np.asarray(np.random.normal(size=(10,5)))
    assert dl.utils.to_float_X(x).dtype == theano.config.floatX


def test_shared_variable():
    x = np.asarray(np.random.normal(size=(10,5)))
    assert isinstance(dl.utils.shared_variable(x), theano.compile.SharedVariable)


def test_format_sec():
    s = 1*24*60*60 + 23*60*60 + 45*60 + 19 + 0.3456
    assert dl.utils.format_sec(s) == '1 d 23 h 45 m 19 s'
    s = 23*60*60 + 45*60 + 19 + 0.3456
    assert dl.utils.format_sec(s) == '23 h 45 m 19 s'
    s = 45*60 + 19 + 0.3456
    assert dl.utils.format_sec(s) == '45 m 19 s'
    s = 19 + 0.3456
    assert dl.utils.format_sec(s) == '19.346 s'
