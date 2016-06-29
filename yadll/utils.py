# -*- coding: UTF-8 -*-
import timeit
from functools import wraps

import numpy as np

import theano

import logging

logger = logging.getLogger(__name__)

floatX = theano.config.floatX
intX = 'int32'
EPSILON = 1e-8


def to_float_X(arr):
    """
    Cast to floatX numpy array

    Parameters
    ----------
    arr: list or numpy array

    Returns
    -------
        numpy array of flotX
    """
    return np.asarray(arr, dtype=floatX)


def shared_variable(value, dtype=floatX, name=None, borrow=True, **kwargs):
    """
    Create a Theano *shared Variable

    Parameters
    ----------
    value:
        value of the shared variable
    dtype : default floatX
        type of the shared variable
    name : string, optional
        shared variable name
    borrow : bool, default is True
        if True shared variable we construct does not get a [deep] copy of value.
        So changes we subsequently make to value will also change our shared variable.

    Returns
    -------
        Theano Shared Variable
    """
    if value is None:
        return None
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, borrow=borrow, **kwargs)


def format_sec(sec):
    """
    format a time

    Parameters
    ----------
    sec : float
        time in seconds

    Returns
    -------
        string :
            formatted time in days, hours, minutes and seconds
    """
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
    """
    decorator that send the execution time of the argument function to the logger

    Parameters
    ----------
    what_to_show : `string`, optional
        message displayed after execution

    """
    def func_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = timeit.default_timer()
            res = func(*args, **kwargs)
            end_time = timeit.default_timer()
            s = end_time - start_time
            try:
                msg = what_to_show + ' ' + args[0].name
            except (AttributeError, IndexError, TypeError):
                msg = what_to_show
            logger.info('%s took %s' % (msg, format_sec(s)))
            return res
        return wrapper
    return func_wrapper

