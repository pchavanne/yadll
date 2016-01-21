# -*- coding: UTF-8 -*-


def sgd_updates(gparams, params, learning_rate):
    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(params, gparams)]
    return updates


def momentum_updates(gparams, params, learning_rate, momentum=0.9):
    # TODO implement this method
    raise NotImplementedError


def nesterov_momentum_updates(gparams, params, learning_rate, momentum=0.9):
    # TODO implement this method
    raise NotImplementedError


def adagrad_updates(gparams, params, learning_rate=1.0, epsilon=0.9):
    # TODO implement this method
    raise NotImplementedError


def rmsprop_updates(gparams, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    # TODO implement this method
    raise NotImplementedError


def adadelta_updates(gparams, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    # TODO implement this method
    raise NotImplementedError


def hessian_free_updates(gparams, parms):
    # TODO implement this method
    raise NotImplementedError
