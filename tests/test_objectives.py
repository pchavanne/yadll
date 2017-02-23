# -*- coding: UTF-8 -*-
import numpy as np
from numpy.testing import assert_allclose

import theano
import theano.tensor as T

import yadll

eps = 1e-4

x_val = np.asarray(np.random.uniform(size=(10, 5)), dtype=yadll.utils.floatX)
x_val /= x_val.sum(axis=1, keepdims=True)
y_val = np.asarray(np.random.uniform(size=(10, 5)), dtype=yadll.utils.floatX)
y_val /= y_val.sum(axis=1, keepdims=True)
x = T.matrix('x')
y = T.matrix('y')


def test_mean_squared_error():
    f = theano.function([x, y], yadll.objectives.mean_squared_error(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(np.square(x_val - y_val), axis=1)
    assert_allclose(actual, desired, rtol=eps)


def test_root_mean_squared_error():
    f = theano.function([x, y], yadll.objectives.root_mean_squared_error(x, y))
    actual = f(x_val, y_val)
    desired = np.sqrt(np.mean(np.square(x_val - y_val), axis=1))
    assert_allclose(actual, desired, rtol=eps)


def test_mean_absolute_error():
    f = theano.function([x, y], yadll.objectives.mean_absolute_error(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(np.abs(x_val - y_val), axis=1)
    assert_allclose(actual, desired, rtol=eps)


def test_binary_hinge_error():
    x_val = np.asarray(np.random.uniform(size=(10, 1)), dtype=yadll.utils.floatX)
    y_val = np.asarray(np.random.binomial(n=1, p=0.5, size=(10, 1)), dtype=yadll.utils.floatX)
    y_val = 2 * y_val - 1
    f = theano.function([x, y], yadll.objectives.binary_hinge_error(x, y))
    actual = f(x_val, y_val)
    desired = np.maximum(1. - x_val * y_val, 0.).flatten()
    assert_allclose(actual, desired, rtol=eps)


def test_categorical_hinge_error():
    x_val = np.asarray(np.random.uniform(size=(10, 1)), dtype=yadll.utils.floatX)
    y_val = np.asarray(np.random.binomial(n=1, p=0.5, size=(10, 1)), dtype=yadll.utils.floatX)
    y_val = 2 * y_val - 1
    f = theano.function([x, y], yadll.objectives.categorical_hinge_error(x, y))
    actual = f(x_val, y_val)
    desired = np.maximum(1. - x_val * y_val, 0.).flatten()
    assert_allclose(actual, desired, rtol=eps)


def test_binary_crossentropy_error():
    x_val = np.asarray(np.random.uniform(size=(10, 1)), dtype=yadll.utils.floatX)
    y_val = np.asarray(np.random.binomial(n=1, p=0.5, size=(10, 1)), dtype=yadll.utils.floatX)
    f = theano.function([x, y], yadll.objectives.binary_crossentropy_error(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(-(y_val * np.log(x_val) + (1 - y_val) * np.log(1 - x_val)), axis=-1)
    assert_allclose(actual, desired, rtol=eps)


def test_categorical_crossentropy_error():
    f = theano.function([x, y], yadll.objectives.categorical_crossentropy_error(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(-np.sum(y_val * np.log(x_val), axis=-1))
    assert_allclose(actual, desired, rtol=eps)


def test_kullback_leibler_divergence():
    f = theano.function([x, y], yadll.objectives.kullback_leibler_divergence(x, y))
    actual = f(x_val, y_val)
    desired = np.sum(y_val * np.log(y_val/x_val), axis=-1)
    assert_allclose(actual, desired, rtol=eps)


def test_binary_accuracy():
    f = theano.function([x, y], yadll.objectives.binary_accuracy(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(np.equal(x_val, np.round(y_val)))
    assert_allclose(actual, desired, rtol=eps)


def test_categorical_accuracy():
    f = theano.function([x, y], yadll.objectives.categorical_accuracy(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(np.equal(np.argmax(x_val, axis=-1), np.argmax(y_val, axis=-1)))
    assert_allclose(actual, desired, rtol=eps)


def test_binary_error():
    f = theano.function([x, y], yadll.objectives.binary_error(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(np.equal(x_val, np.round(y_val)))
    assert_allclose(actual, desired, rtol=eps)


def test_categorical_error():
    f = theano.function([x, y], yadll.objectives.categorical_error(x, y))
    actual = f(x_val, y_val)
    desired = np.mean(np.not_equal(np.argmax(x_val, axis=-1), np.argmax(y_val, axis=-1)))
    assert_allclose(actual, desired, rtol=eps)
