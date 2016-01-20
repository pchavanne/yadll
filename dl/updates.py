#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "Philippe Chavanne"


def sgd_updates(gparams, params, learning_rate):
    updates = [(param, param - learning_rate * gparam)
               for param, gparam in zip(params, gparams)]
    return updates


def momentum_updates(gparams, params, learning_rate, momentum=0.9):
    # TODO implement this method
    pass


def nesterov_momentum_updates(gparams, params, learning_rate, momentum=0.9):
    # TODO implement this method
    pass


def adagrad_updates(gparams, params, learning_rate=1.0, epsilon=0.9):
    # TODO implement this method
    pass


def rmsprop_updates(gparams, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    # TODO implement this method
    pass


def adadelta_updates(gparams, params, learning_rate=1.0, rho=0.95, epsilon=1e-6):
    # TODO implement this method
    pass


def hessian_free_updates(gparams, parms):
    # TODO implement this method
    pass
