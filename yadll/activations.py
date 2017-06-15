# -*- coding: UTF-8 -*-
"""
Activation functions
"""

import theano.tensor as T


def get_activation(activator):
    r"""
    Call an activation function from an activator object

    Parameters
    ----------
    activator : `activator`
            an activator is an activation function, a tuple of (activation function, dict of args),
            the name of the activation function as a str or a tuple (name of function, dict of args)
            example : activator = tanh  or activator = (elu, {'alpha':0.5})
                   or activator = 'tanh'  or activator = ('elu', {'alpha':0.5})

    Returns
    -------
        an activation function
    """
    if isinstance(activator, tuple):
        kwargs = activator[1]
        if isinstance(activator[0], str):
            func = eval(activator[0])
        else:
            func = activator[0]
    else:
        kwargs = {}
        if isinstance(activator, str):
            func = eval(activator)
        else:
            func = activator

    def activation(x):
        return func(x, **kwargs)

    return activation


def activation_to_conf(activation):
    func_name = activation.__closure__[0].cell_contents.__name__
    kwargs = activation.__closure__[1].cell_contents
    if kwargs == {}:
        conf = func_name
    else:
        conf = (func_name, kwargs)
    return conf


def linear(x):
    r"""Linear activation function
    :math: `\varphi(x) = x`

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.

    Returns
    -------
    symbolic tensor
        The output of the identity applied to the activation `x`.

    """
    return x


def sigmoid(x):
    r"""Sigmoid function
    :math:`\varphi(x) = \frac{1}{1 + \exp{-x}}`

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.

    Returns
    -------
    symbolic tensor of value in [0, 1]
        The output of the sigmoid function applied to the activation `x`.

    """
    return T.nnet.sigmoid(x)


def ultra_fast_sigmoid(x):
    r"""Ultra fast Sigmoid function return an approximated standard sigmoid
    :math:`\varphi(x) = \frac{1}{1+\exp{-x}}`

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.

    Returns
    -------
    symbolic tensor of value in [0, 1]
        The output of the sigmoid function applied to the activation `x`.

    Notes
    _____
    Use the Theano flag optimizer_including=local_ultra_fast_sigmoid to use
    ultra_fast_sigmoid systematically instead of sigmoid.

    """
    return T.nnet.ultra_fast_sigmoid(x)


def tanh(x):
    r"""Tanh activation function
    :math:`\varphi(x) = \tanh(x)`

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.

    Returns
    -------
    symbolic tensor of value in [-1, 1]
        The output of the tanh function applied to the activation `x`.

    """
    return T.tanh(x)


def softmax(x):
    r"""Softmax activation function
    :math:`\varphi(x)_j = \frac{\exp{x_j}}{\sum_{k=1}^K \exp{x_k}}`

    where :math:`K` is the total number of neurons in the layer. This
    activation function gets applied row-wise.

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.

    Returns
    -------
    symbolic tensor where the sum of the row is 1 and each single value is in [0, 1]
        The output of the softmax function applied to the activation `x`.

    """
    return T.nnet.softmax(x)


def softplus(x):
    r"""Softplus activation function :math:`\varphi(x) = \log{1 + \exp{x}}`

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.

    Returns
    -------
    symbolic tensor
        The output of the softplus function applied to the activation `x`.

    """
    return T.nnet.softplus(x)


def relu(x, alpha=0):
    r"""Rectified linear unit activation function
    :math:`\varphi(x) = \max{x, \alpha * x}`

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.
    alpha : scalar or tensor, optional
        Slope for negative input, usually between 0 and 1. The default value
        of 0 will lead to the standard rectifier, 1 will lead to
        a linear activation function, and any value in between will give a
        leaky rectifier. A shared variable (broadcastable against `x`) will
        result in a parameterized rectifier with learnable slope(s).

    Returns
    -------
    symbolic tensor
        Element-wise rectifier applied to the activation `x`.

    Notes
    -----
    This is numerically equivalent to ``T.switch(x > 0, x, alpha * x)``
    (or ``T.maximum(x, alpha * x)`` for ``alpha < 1``), but uses a faster
    formulation or an optimized Op, so we encourage to use this function.

    References
    ----------
    .. [1] Xavier Glorot, Antoine Bordes and Yoshua Bengio (2011):
           Deep sparse rectifier neural networks. AISTATS.
           http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf

    """
    return T.nnet.relu(x, alpha)


def elu(x, alpha=1):
    r"""
    Compute the element-wise exponential linear activation function.

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.
    alpha : scalar


    Returns
    -------
    symbolic tensor
        Element-wise exponential linear activation function applied to `x`.

    References
    -----
    .. [1] Djork-Arne Clevert,  Thomas Unterthiner, Sepp Hochreiter
        "Fast and Accurate Deep Network Learning by
        Exponential Linear Units (ELUs)" <http://arxiv.org/abs/1511.07289>`.
    """
    return T.nnet.elu(x, alpha)


def selu(x):
    r"""
    Compute the element-wise scaled exponential linear activation function.

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.

    Returns
    -------
    symbolic tensor
        Element-wise scaled exponential linear activation function applied to `x`.

    References
    -----
    .. [1] https://arxiv.org/pdf/1706.02515.pdf
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)
