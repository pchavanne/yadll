# -*- coding: UTF-8 -*-
import numpy as np

from .utils import shared_variable
from .activation import *

np_rng = np.random.RandomState(1234)

# init_obj = glorot_uniform  or init_obj = (glorot_uniform, {'gain':tanh, 'borrow':False})


def initializer(init_obj, shape, name, fan=None):
    if not isinstance(init_obj, tuple):
        return init_obj(shape, name=name, fan=fan)
    else:
        return init_obj[0](shape, name=name, fan=fan, **init_obj[1])


def constant(shape, value=0.0, name=None, borrow=True, **kwargs):
    return shared_variable(np.ones(shape=shape) * value,
                           name=name, borrow=borrow)


def uniform(shape, scale=0.5, name=None, borrow=True, **kwargs):
    if not isinstance(scale, tuple):
        scale = (-scale, scale)      # (low, high)
    return shared_variable(np_rng.uniform(low=scale[0], high=scale[1], size=shape),
                           name=name, borrow=borrow)


def normal(shape, scale=0.5, name=None, borrow=True, **kwargs):
    return shared_variable(np_rng.normal(loc=0.0, scale=scale, size=shape),
                           name=name, borrow=borrow)


def glorot_uniform(shape, gain=1.0, name=None, fan=None, borrow=True):
    if fan:
        fan_in, fan_out = fan
    else:
        fan_in, fan_out = shape
    if gain == tanh:
        gain = 1.
    if gain == sigmoid:
        gain = 4.
    scale = gain * np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, scale, name, borrow)


def glorot_normal(shape, gain=1, name=None, fan=None, borrow=True):
    if fan:
        fan_in, fan_out = fan
    else:
        fan_in, fan_out = shape
    if gain == tanh:
        gain = 1.
    if gain == sigmoid:
        gain = 4.
    scale = gain * np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, scale, name, borrow)


def He_uniform(shape, name=None, fan=None, borrow=True):
    scale = np.sqrt(6. / shape[0])
    return uniform(shape, scale, name, borrow)


def He_normal(shape, name=None, fan=None, borrow=True):
    scale = np.sqrt(2. / shape[0])
    return normal(shape, scale, name, borrow)


def orthogonal(shape, gain=1, name=None, fan=None, borrow=True):
    if gain == relu:
        gain = np.sqrt(2)
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, size=flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return shared_variable(gain * q, name=name, borrow=borrow)