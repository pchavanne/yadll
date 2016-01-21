# -*- coding: UTF-8 -*-

from collections import OrderedDict

from utils import *
from .init import *
from . import activation


class Layer(object):
    def __init__(self, incoming, params, name=None):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        self.params = [ for k, v in params]

    @property
    def output_shape(self):
        return self.get_output_shape(self.input_shape)

    def get_output_shape(self, input_shape):
        raise NotImplementedError

    def get_output(self, input, **kwargs):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, incoming, nb_units, name=None,
                 params={'W': glorot_uniform, 'b': constant},
                 activation=tanh):
        super(Dense, self).__init__(incoming, name)
        self.init =

    def get_output_shape(self, input_shape):
        raise NotImplementedError

    def get_output(self, input, **kwargs):
        raise NotImplementedError