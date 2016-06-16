# -*- coding: UTF-8 -*-
"""
Activation functions
"""

import theano.tensor as T


def sigmoid(x):
    """Sigmoid function
    :math:`\\varphi(x) = \\frac{1}{1 + e^{-x}}`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32 in [0, 1]
        The output of the sigmoid function applied to the activation.

    """
    return T.nnet.sigmoid(x)


def ultra_fast_sigmoid(x):
    """Ultra fast Sigmoid function return an approximated standard sigmoid
    :math:`\\varphi(x) = \\frac{1}{1 + e^{-x}}`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32 in [0, 1]
        The output of the sigmoid function applied to the activation.

    Notes
    _____

    """
    return T.nnet.ultra_fast_sigmoid(x)


def tanh(x):
    """Tanh activation function
    :math:`\\varphi(x) = \\tanh(x)`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32 in [-1, 1]
        The output of the tanh function applied to the activation.

    """
    return T.tanh(x)


def softmax(x):
    """Softmax activation function
    :math:`\\varphi(\\mathbf{x})_j =
    \\frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`
    where :math:`K` is the total number of neurons in the layer. This
    activation function gets applied row-wise.

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32 where the sum of the row is 1 and each single value is in [0, 1]
        The output of the softmax function applied to the activation.

    """
    return T.nnet.softmax(x)


def softplus(x):
    """Softplus activation function :math:`\\varphi(x) = \\log(1 + e^x)`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32
        The output of the softplus function applied to the activation.

    """
    return T.nnet.softplus(x)


def relu(x, alpha=0):
    """Rectified linear unit activation function
    :math:`\\varphi(x) = \\max(0, x)`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32
        The output of the rectify function applied to the activation.

    Notes
    -----
    This is numerically equivalent to ``T.switch(x > 0, x, alpha * x)``
    (or ``T.maximum(x, alpha * x)`` for ``alpha < 1``), but uses a faster
    formulation or an optimized Op, so we encourage to use this function.

    """
    return T.nnet.relu(x, alpha)


def linear(x):
    """Linear activation function
    :math:`\\varphi(x) = x`

    Parameters
    ----------
    x : float32
        The activation (the summed, weighted input of a neuron).

    Returns
    -------
    float32
        The output of the identity applied to the activation.

    """
    return x

