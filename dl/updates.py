# -*- coding: UTF-8 -*-

import theano.tensor as T

from .utils import *


def sgd_updates(cost, params, learning_rate):
    gparams = [T.grad(cost, param) for param in params]
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]
    return updates


def momentum_updates(cost, params, learning_rate, momentum=0.9):
    updates = []
    for param in params:
        param_update = shared_variable(np.zeros(param.get_value().shape))
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update, momentum * param_update + (1. - momentum) * T.grad(cost, param)))
    return updates


def nesterov_momentum_updates(cost, params, learning_rate, momentum=0.9):
    updates = []
    for param in params:
        param_update = shared_variable(np.zeros(param.get_value().shape))
        updates.append((param, param - learning_rate * param_update))
        eval_param = param + momentum * param_update
        updates.append((param_update, momentum * param_update + (1. - momentum) * T.grad(cost, eval_param)))
    return updates


def adagrad_updates(cost, params, learning_rate=1.0, epsilon=0.9):
    # TODO implement this method
    raise NotImplementedError


def adadelta_updates(cost, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    # TODO implement this method
    raise NotImplementedError


def rmsprop_updates(cost, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    # TODO implement this method
    raise NotImplementedError


def hessian_free_updates(cost, parms):
    # TODO implement this method
    raise NotImplementedError
