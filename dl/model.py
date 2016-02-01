# -*- coding: UTF-8 -*-

import timeit

import theano.tensor as T

from .layers import *


class Network(object):
    def __init__(self, name=None, layers=None):
        if not layers:
            self.layers = []
        else:
            for layer in layers:
                self.add(layer)
        self.params = []
        self.reguls = 0

    def add(self, layer):
        self.layers.append(layer)
        self.params.extend(layer.params)
        self.reguls += layer.reguls

    def params(self):
        return self.params

    def reguls(self):
        return self.reguls

    def get_output(self, **kwargs):
        return self.layers[-1].get_output(**kwargs)


class Model(object):
    def __init__(self, network, data, name=None, file=None):
        self.network = network
        self.data = data             # data [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
        self.name = name
        self.file = file
        self.index = T.iscalar()     # index to a [mini]batch
        self.x = T.matrix('x')       # the input data is presented as a matrix
        if data.train_set_y.ndim == 1:
            self.y = T.ivector('y')      # the output labels are presented as 1D vector of[int] labels
        else:
            self.y = T.matrix('y')

    def pretrain(self):
        for layer in self.network.layers:
            if isinstance(layer, UnsupervisedLayer):
                layer.unsupervised_training(self.data.train_set_x)

    def train(self):
        pass


