# -*- coding: UTF-8 -*-

import numpy as np
from numpy.testing import assert_allclose

import theano
import theano.tensor as T

import dl

x_val = np.asarray(np.random.uniform(size=(10, 5)), dtype=dl.utils.floatX)
x_val /= x_val.sum(axis=1)[:, np.newaxis]
y_val = np.asarray(np.random.uniform(size=(10, 5)), dtype=dl.utils.floatX)
y_val /= y_val.sum(axis=1)[:, np.newaxis]
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


def test_categorical_crossentropy():
    f = theano.function([x, y], dl.objectives.categorical_crossentropy(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(- np.sum(x_val * np.log(y_val) + (1 - x_val) * np.log(1 - y_val), axis=1))
    assert_allclose(actual, desired, rtol=1e-2)

