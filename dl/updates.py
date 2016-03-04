# -*- coding: UTF-8 -*-
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
        velocity = shared_variable(np.zeros(param.get_value(borrow=True).shape))
        p = momentum * velocity + updates[param]
        updates[velocity] = p - param
        updates[param] = p
    return updates


def nesterov_momentum(cost, params, learning_rate=0.1, momentum=0.9, **kwargs):
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum
    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + momentum * velocity - learning_rate * gradient``
    """
    updates = sgd(cost, params, learning_rate)
    for param in params:
        velocity = shared_variable(np.zeros(param.get_value(borrow=True).shape))
        p = momentum * velocity + updates[param] - param
        updates[velocity] = p
        updates[param] = momentum * p + updates[param]
    return updates


def adagrad(cost, params, learning_rate=1.0, epsilon=1e-6, **kwargs):
    """Adaptive Gradient Descent
    Scale learning rates by dividing with the square root of accumulated
    squared gradients
    """
    gparams = T.grad(cost, params)
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        accu = shared_variable(np.zeros(param.get_value(borrow=True).shape))
        accu_new = accu + gparam ** 2
        updates[accu] = accu_new
        updates[param] = param - learning_rate * gparam / T.sqrt(accu_new + epsilon)
    return updates


def adadelta(cost, params, learning_rate=1.0, rho=0.95, epsilon=1e-6, **kwargs):
    """Adadelta Gradient Descent
    Scale learning rates by a the ratio of accumulated gradients to accumulated
    step sizes
    """
    gparams = T.grad(cost, params)
    updates = OrderedDict()

    for param, gparam in zip(params, gparams):
        accu = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)
        delta_accu = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)

        # update accu (as in rmsprop)
        accu_new = rho * accu + (1 - rho) * gparam ** 2
        updates[accu] = accu_new

        # compute parameter update, using the 'old' delta_accu
        update = (gparam * T.sqrt(delta_accu + epsilon) /
                  T.sqrt(accu_new + epsilon))
        updates[param] = param - learning_rate * update

        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = rho * delta_accu + (1 - rho) * update ** 2
        updates[delta_accu] = delta_accu_new

    return updates


def rmsprop(cost, params, learning_rate=1.0, rho=0.9, epsilon=1e-6, **kwargs):
    """RMSProp updates
    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients
    """
    gparams = T.grad(cost, params)
    updates = OrderedDict()
    for param, gparam in zip(params, gparams):
        accu = shared_variable(np.zeros(param.get_value(borrow=True).shape), broadcastable=param.broadcastable)
        accu_new = rho * accu + (1 - rho) * gparam ** 2
        updates[accu] = accu_new
        updates[param] = param - learning_rate * gparam / T.sqrt(accu_new + epsilon)
    return updates


def hessian_free(cost, parms, **kwargs):
    # TODO implement hessian_free method
    raise NotImplementedError
