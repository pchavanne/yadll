# -*- coding: UTF-8 -*-
"""
Updates functions

Arguments
---------
cost : cost function
    The cost function that will be minimised during training
params : list of parameters
    The list of all the weights of the network that will be modified

"""

from collections import OrderedDict

import theano.tensor as T

from .utils import *


def sgd(cost, params, learning_rate=0.1, **kwargs):
    """Stochastic Gradient Descent (SGD) updates
    * ``param := param - learning_rate * gradient``
    """
    gparams = T.grad(cost, params)
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        updates[param] = param - learning_rate * gparam
    return updates


def momentum(cost, params, learning_rate=0.1, momentum=0.9, **kwargs):
    """Stochastic Gradient Descent (SGD) updates with momentum
    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + velocity``
    """
    updates = sgd(cost, params, learning_rate)
    for param in params:
        velocity = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)
        p = momentum * velocity + updates[param]
        updates[velocity] = p - param
        updates[param] = p
    return updates


def nesterov_momentum(cost, params, learning_rate=0.1, momentum=0.9, **kwargs):
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum
    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + momentum * velocity - learning_rate * gradient``

    References
    ----------
    .. [1] https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617
    """
    updates = sgd(cost, params, learning_rate)
    for param in params:
        velocity = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)
        p = momentum * velocity + updates[param] - param
        updates[velocity] = p
        updates[param] += momentum * p
    return updates


def adagrad(cost, params, learning_rate=1.0, epsilon=1e-6, **kwargs):
    """Adaptive Gradient Descent
    Scale learning rates by dividing with the square root of accumulated
    squared gradients

    References
    ----------
    .. [1] http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    """
    gparams = T.grad(cost, params)
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        accu = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)
        accu_new = accu + gparam ** 2
        updates[accu] = accu_new
        updates[param] = param - learning_rate * gparam / T.sqrt(accu_new + epsilon)
    return updates


def rmsprop(cost, params, learning_rate=0.01, rho=0.9, epsilon=1e-6, **kwargs):
    """RMSProp updates
    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients
    """
    gparams = T.grad(cost, params)
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        accu = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)
        accu_new = rho * accu + (1. - rho) * gparam ** 2
        updates[accu] = accu_new
        updates[param] = param - learning_rate * gparam / T.sqrt(accu_new + epsilon)
    return updates


def adadelta(cost, params, learning_rate=1.0, rho=0.95, epsilon=1e-6, **kwargs):
    """Adadelta Gradient Descent
    Scale learning rates by a the ratio of accumulated gradients to accumulated
    step sizes

    References
    ----------
    .. [1] https://arxiv.org/pdf/1212.5701v1.pdf
    """
    gparams = T.grad(cost, params)
    updates = OrderedDict()

    for param, gparam in zip(params, gparams):
        accu = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)
        delta_accu = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)

        # update accu (as in rmsprop)
        accu_new = rho * accu + (1. - rho) * gparam ** 2
        updates[accu] = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = (gparam * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        updates[param] = param - learning_rate * update

        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (1. - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates


def adam(cost, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6, **kwargs):
    """Adam Gradient Descent
    Scale learning rates by Adaptive moment estimation

    References
    ----------
    .. [1] https://arxiv.org/pdf/1412.6980v8.pdf
    """
    gparams = T.grad(cost, params)
    updates = OrderedDict()
    t = theano.shared(floatX(0.))
    t_t = 1. + t
    l_r_t = learning_rate * T.sqrt(1. - beta2 ** t_t) / (1. - beta1 ** t_t)
    for param, gparam in zip(params, gparams):
        m = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)
        v = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)
        m_t = beta1 * m + (1. - beta1) * gparam
        v_t = beta2 * v + (1. - beta2) * T.sqr(gparam)
        updates[m] = m_t
        updates[v] = v_t
        updates[param] = param - l_r_t * m_t / (T.sqrt(v_t) + epsilon)
    updates[t] = t_t
    return updates


def adamax(cost, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6, **kwargs):
    """Adam Gradient Descent
    Scale learning rates by adaptive moment estimation

    References
    ----------
    .. [1] https://arxiv.org/pdf/1412.6980v8.pdf
    """
    gparams = T.grad(cost, params)
    updates = OrderedDict()
    t = theano.shared(floatX(0.))
    t_t = 1. + t
    l_r_t = learning_rate / (1. - beta1 ** t_t)
    for param, gparam in zip(params, gparams):
        m = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)
        u = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)
        m_t = beta1 * m + (1. - beta1) * gparam
        u_t = T.maximum(beta2 * u, abs(gparam))
        updates[m] = m_t
        updates[u] = u_t
        updates[param] = param - l_r_t * m_t / (u_t + epsilon)
    updates[t] = t_t
    return updates


def nadam(cost, params, learning_rate=1.0, rho=0.95, epsilon=1e-6, **kwargs):
    """Adam Gradient Descent
    Nesterov Momentum in Adam

    References
    ----------
    .. [1] http://cs229.stanford.edu/proj2015/054_report.pdf
    """
    # TODO implement nadam method
    raise NotImplementedError
    gparams = T.grad(cost, params)
    updates = OrderedDict()

    return updates


def hessian_free(cost, parms, **kwargs):
    """
    Hessian Free optimization

    References
    ----------
    .. [1] http://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf
    .. [2] http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
    .. [3] http://www.cs.utoronto.ca/~ilya/pubs/2011/HF-RNN.pdf
    .. [4] http://olivier.chapelle.cc/pub/precond.pdf
    .. [5] http://www.cs.toronto.edu/~rkiros/papers/shf13.pdf
    """
    # TODO implement hessian_free method
    raise NotImplementedError
