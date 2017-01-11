# -*- coding: UTF-8 -*-
import numpy as np

from .utils import shared_variable
from .activations import *

np_rng = np.random.RandomState(1234)

# init_obj = glorot_uniform  or init_obj = (glorot_uniform, {'gain':tanh, 'borrow':False})


def initializer(init_obj, shape, name, **kwargs):
    """
    Call an Initializer from an init_obj

    Parameters
    ----------
    init_obj : `init_obj`
            an init_obj is an initializer function or the tuple of (initializer function, dict of args)
            example : init_obj = glorot_uniform  or init_obj = (glorot_uniform, {'gain':tanh, 'borrow':False})
    shape : `tuple` or int
        shape of the return shared variables
    Returns
    -------
        Initialized shared variables
    """
    if not isinstance(init_obj, tuple):
        return init_obj(shape, name=name, **kwargs)
    else:
        kwargs.update(init_obj[1])
        return init_obj[0](shape, name=name, **kwargs)


def constant(shape, value=0.0, name=None, borrow=True, **kwargs):
    """
    Initialize all the weights to a constant value

    Parameters
    ----------
    shape
    scale
    """
    return shared_variable(np.ones(shape=shape) * value,
                           name=name, borrow=borrow, **kwargs)


def uniform(shape, scale=0.5, name=None, borrow=True, **kwargs):
    """
    Initialize all the weights from the uniform distribution

    Parameters
    ----------
    shape
    scale
    name
    borrow
    kwargs

    Returns
    -------

    """
    if not isinstance(scale, tuple):
        scale = (-scale, scale)      # (low, high)
    return shared_variable(np_rng.uniform(low=scale[0], high=scale[1], size=shape),
                           name=name, borrow=borrow, **kwargs)


def normal(shape, scale=0.5, name=None, borrow=True, **kwargs):
    """
    Initialize all the weights from the normal distribution

    Parameters
    ----------
    shape
    scale
    name
    borrow
    kwargs

    Returns
    -------

    """
    return shared_variable(np_rng.normal(loc=0.0, scale=scale, size=shape),
                           name=name, borrow=borrow, **kwargs)


def glorot_uniform(shape, gain=1.0, name=None, fan=None, borrow=True, **kwargs):
    """
    Initialize all the weights from the uniform distribution with glorot scaling

    Parameters
    ----------
    shape
    gain
    name
    fan
    borrow
    kwargs

    Returns
    -------

    References
    ----------
    .. [1] http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

    """
    if fan:
        fan_in, fan_out = fan
    else:
        fan_in, fan_out = shape
    if gain == tanh:
        gain = 1.
    if gain == sigmoid:
        gain = 4.
    scale = gain * np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, scale, name, borrow, **kwargs)


def glorot_normal(shape, gain=1, name=None, fan=None, borrow=True, **kwargs):
    """
    Initialize all the weights from the normal distribution with glorot scaling

    Parameters
    ----------
    shape
    gain
    name
    fan
    borrow
    kwargs

    Returns
    -------

    References
    ----------
    .. [1] http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

    """
    if fan:
        fan_in, fan_out = fan
    else:
        fan_in, fan_out = shape
    if gain == tanh:
        gain = 1.
    if gain == sigmoid:
        gain = 4.
    scale = gain * np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, scale, name, borrow, **kwargs)


def He_uniform(shape, name=None, borrow=True, **kwargs):
    scale = np.sqrt(6. / shape[0])
    return uniform(shape, scale, name, borrow, **kwargs)


def He_normal(shape, name=None, borrow=True, **kwargs):
    scale = np.sqrt(2. / shape[0])
    return normal(shape, scale, name, borrow, **kwargs)


def orthogonal(shape, gain=1, name=None, borrow=True, **kwargs):
    """
    Orthogonal initialization for Recurrent Networks

    Orthogonal initialization solve the vanishing/exploding gradient for
    recurrent network.

    Parameters
    ----------
    shape
    gain
    name
    borrow
    kwargs

    Returns
    -------

    References
    ----------

    .. [1] http://smerity.com/articles/2016/orthogonal_init.html
    """
    if gain == relu:
        gain = np.sqrt(2)
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, size=flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return shared_variable(gain * q, name=name, borrow=borrow, **kwargs)
