# -*- coding: UTF-8 -*-

import pytest

import numpy as np
from numpy.testing import assert_allclose

import theano
import theano.tensor as T

import dl

x_val = np.asarray([[-10, -5, -1, -0.9, -0.1, 0, 0.1, 0.9, 1, 5, 10]],
                   dtype=dl.utils.floatX)


def test_sigmoid():
    x = T.matrix('x')
    f = theano.function([x], dl.activation.sigmoid(x))
    actual = f(x_val)
    desired = 1 / (1 + np.exp(-x_val))
    assert_allclose(actual, desired, rtol=1e-5)


def test_tanh():
    x = T.matrix('x')
    f = theano.function([x], dl.activation.tanh(x))
    actual = f(x_val)
    desired = np.tanh(x_val)
    assert_allclose(actual, desired, rtol=1e-5)


def test_softmax():
    x = T.matrix('x')
    f = theano.function([x], dl.activation.softmax(x))
    actual = f(x_val)
    desired = np.exp(x_val) / np.exp(x_val).sum()
    assert_allclose(actual, desired, rtol=1e-5)


def test_softplus():
    x = T.matrix('x')
    f = theano.function([x], dl.activation.softplus(x))
    actual = f(x_val)
    desired = np.log(1 + np.exp(x_val))
    assert_allclose(actual, desired, rtol=1e-5)


def test_relu():
    x = T.matrix('x')
    f = theano.function([x], dl.activation.relu(x))
    actual = f(x_val)
    desired = 1 / 1 + np.exp(-x_val)
    assert_allclose(actual, desired, rtol=1e-5)


def test_linear(x):
    x = T.matrix('x')
    f = theano.function([x], dl.activation.linear(x))
    actual = f(x_val)
    desired = x_val
    assert_allclose(actual, desired, rtol=1e-5)

