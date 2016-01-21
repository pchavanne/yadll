# -*- coding: UTF-8 -*-

from collections import OrderedDict

from utils import *
from .init import *
from . import activation


class Layer(object):
    def __init__(self, incoming, name=None):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        self.params = []

    @property
    def output_shape(self):
        return self.input_shape

    def get_output(self, input, **kwargs):
        raise NotImplementedError


class Input_Layer(Layer):
    def __init__(self, shape, input_var=None, name=None):
        self.shape = shape
        self.input = input_var
        self.name = name

    @property
    def output_shape(self):
        return self.shape

    def get_output(self, input, **kwargs):
        return self.input


class Dense_Layer(Layer):
    def __init__(self, incoming, nb_units, name=None,
                 W=glorot_uniform, b=(constant, {'value':0.0}),
                 activation=tanh):
        super(Dense, self).__init__(incoming, name)
        self.shape = (incoming.output_shape, nb_units)
        self.W = initializer(W, shape=self.shape, name='W')
        self.params.append(self.W)
        self.b = initializer(b, shape=(self.shape[1],), name='b')
        self.params.append(self.b)
        self.activation = activation

    @property
    def output_shape(self):
        return self.shape

    def get_output(self, input, **kwargs):
        X = self.input_layer.get_output()
        return self.activation(T.dot(X, self.W) + self.b)

