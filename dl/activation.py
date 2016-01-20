#!/usr/bin/env python
# -*- coding: UTF-8 -*-

__author__ = "Philippe Chavanne"

import theano.tensor as T


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def softmax(x):
    return T.nnet.softmax(x)


def relu(x):
    return T.nnet.relu(x)