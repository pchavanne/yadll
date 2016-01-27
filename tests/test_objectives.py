# -*- coding: UTF-8 -*-

import numpy as np
from numpy.testing import assert_allclose

import theano
import theano.tensor as T

import dl

x_val = np.asarray(np.random.uniform(size=(10, 5)), dtype=dl.utils.floatX)
x_val /= x_val.sum(axis=1, keepdims=True)
y_val = np.asarray(np.random.uniform(size=(10, 5)), dtype=dl.utils.floatX)
y_val /= y_val.sum(axis=1, keepdims=True)
x = T.matrix('x')
y = T.matrix('y')


def test_mean_squared_error():
    f = theano.function([x, y], dl.objectives.mean_squared_error(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(np.square(x_val - y_val), axis=1)
    assert_allclose(actual, desired, rtol=1e-5)


def test_root_mean_squared_error():
    f = theano.function([x, y], dl.objectives.root_mean_squared_error(x, y))
    actual = f(x_val, y_val)
    desired = np.sqrt(np.mean(np.square(x_val - y_val), axis=1))
    assert_allclose(actual, desired, rtol=1e-5)


def test_mean_absolute_error():
    f = theano.function([x, y], dl.objectives.mean_absolute_error(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(np.abs(x_val - y_val), axis=1)
    assert_allclose(actual, desired, rtol=1e-5)


def test_hinge():
    x_val = np.asarray(np.random.uniform(size=(10, 1)), dtype=dl.utils.floatX)
    y_val = np.asarray(np.random.binomial(n=1,p=0.5, size=(10, 1)), dtype=dl.utils.floatX)
    y_val = 2 * y_val - 1
    f = theano.function([x, y], dl.objectives.hinge(x, y))
    actual = f(x_val, y_val)
    desired = np.maximum(1. - x_val * y_val, 0.).flatten()
    assert_allclose(actual, desired, rtol=1e-5)


def test_binary_crossentropy():
    x_val = np.asarray(np.random.uniform(size=(10, 1)), dtype=dl.utils.floatX)
    y_val = np.asarray(np.random.binomial(n=1,p=0.5, size=(10, 1)), dtype=dl.utils.floatX)
    f = theano.function([x, y], dl.objectives.binary_crossentropy(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(-(y_val * np.log(x_val) + (1 - y_val) * np.log(1 - x_val)), axis=-1)
    assert_allclose(actual, desired, rtol=1e-5)


def test_categorical_crossentropy():
    f = theano.function([x, y], dl.objectives.categorical_crossentropy(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(-np.sum(y_val * np.log(x_val), axis=-1))
    assert_allclose(actual, desired, rtol=1e-5)

