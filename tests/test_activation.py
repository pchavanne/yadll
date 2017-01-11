# -*- coding: UTF-8 -*-
import numpy as np
from numpy.testing import assert_allclose

import theano
import theano.tensor as T

import yadll

x_val = np.asarray([[-10, -5, -1, -0.9, -0.1, 0, 0.1, 0.9, 1, 5, 10]],
                   dtype=yadll.utils.floatX)


def test_sigmoid():
    x = T.matrix('x')
    f = theano.function([x], yadll.activations.sigmoid(x))
    actual = f(x_val)
    desired = 1 / (1 + np.exp(-x_val))
    assert_allclose(actual, desired, rtol=1e-5)


def test_ultra_fast_sigmoid():
    x = T.matrix('x')
    f = theano.function([x], yadll.activations.ultra_fast_sigmoid(x))
    actual = f(x_val)
    desired = 1 / (1 + np.exp(-x_val))
    assert_allclose(actual, desired, rtol=0, atol=1e-1)


def test_tanh():
    x = T.matrix('x')
    f = theano.function([x], yadll.activations.tanh(x))
    actual = f(x_val)
    desired = np.tanh(x_val)
    assert_allclose(actual, desired, rtol=1e-5)


def test_softmax():
    x = T.matrix('x')
    f = theano.function([x], yadll.activations.softmax(x))
    actual = f(x_val)
    desired = np.exp(x_val) / np.exp(x_val).sum()
    assert_allclose(actual, desired, rtol=1e-5)


def test_softplus():
    x = T.matrix('x')
    f = theano.function([x], yadll.activations.softplus(x))
    actual = f(x_val)
    desired = np.log(1 + np.exp(x_val))
    assert_allclose(actual, desired, rtol=1e-3)


def test_relu():
    x = T.matrix('x')
    f = theano.function([x], yadll.activations.relu(x))
    actual = f(x_val)
    desired = x_val * (x_val > 0)
    assert_allclose(actual, desired, rtol=1e-5)
    x = T.matrix('x')
    alpha = 0.5
    f = theano.function([x], yadll.activations.relu(x, alpha))
    actual = f(x_val)
    desired = x_val * (x_val > 0) + alpha * x_val * (x_val < 0)
    assert_allclose(actual, desired, rtol=1e-5)


def test_linear():
    x = [0, -1, 1, 3.2, 1e-7, np.inf, True, None, 'foo']
    actual = yadll.activations.linear(x)
    desired = x
    assert actual == desired

