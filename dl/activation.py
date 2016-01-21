# -*- coding: UTF-8 -*-


import theano.tensor as T


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def softmax(x):
    return T.nnet.softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def relu(x, alpha=0):
    return T.nnet.relu(x, alpha)


def linear(x):
    return x

