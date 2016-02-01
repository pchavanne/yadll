# -*- coding: UTF-8 -*-
import timeit
from functools import wraps

import numpy as np

import theano


floatX = theano.config.floatX
intX = 'int32'
EPSILON = 1e-8


def to_float_X(arr):
    return np.asarray(arr, dtype=floatX)


def shared_variable(value, dtype=floatX, name=None, borrow=True):
    if value is None:
        return None
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, borrow=True)


def format_sec(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:
        return '%d d %02d h %02d m %02d s' % (d, h, m, s)
    if h:
        return '%02d h %02d m %02d s' % (h, m, s)
    if m:
        return '%02d m %02d s' % (m, s)
    return '%.3f s' % s


def timer(what_to_show="Function execution"):
    def func_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = timeit.default_timer()
            res = func(*args, **kwargs)
            end_time = timeit.default_timer()
            s = end_time - start_time
            try:
                msg = what_to_show + ' ' + args[0].name
            except (AttributeError, IndexError):
                msg = what_to_show
            print '%s took %s' % (msg, format_sec(s))
            return res
        return wrapper
    return func_wrapper

