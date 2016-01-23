# -*- coding: UTF-8 -*-

from .init import *
from .utils import *

from theano.tensor.shared_randomstreams import RandomStreams

T_rng = RandomStreams(np_rng.randint(2 ** 30))


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

    def get_output(self, **kwargs):
        raise NotImplementedError


class InputLayer(Layer):
    def __init__(self, shape, input_var=None, name=None):
        super(InputLayer, self).__init__(shape, name)
        self.input = input_var

    def get_output(self, **kwargs):
        return self.input


class DenseLayer(Layer):
    def __init__(self, incoming, nb_units, name=None,
                 W=glorot_uniform, b=(constant, {'value':0.0}),
                 activation=tanh):
        super(DenseLayer, self).__init__(incoming, name)
        self.shape = (self.input_shape[1], nb_units)
        self.W = initializer(W, shape=self.shape, name='W')
        self.params.append(self.W)
        self.b = initializer(b, shape=(self.shape[1],), name='b')
        self.params.append(self.b)
        self.activation = activation

    @property
    def output_shape(self):
        return self.input_shape[0], self.shape[1]

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return self.activation(T.dot(X, self.W) + self.b)


class LogisticRegression(DenseLayer):
    def __init__(self, incoming, nb_class, name=None,
                 W=constant, b=constant, activation=softmax):
        super(LogisticRegression, self).__init__(incoming, nb_class, name=name,
                                                 W=W, b=b, activation=activation)


class Dropout(Layer):
    def __init__(self, incoming, corruption_level=0.5, name=None):
        super(Dropout, self).__init__(incoming, name)
        self.p = 1 - corruption_level

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.p > 0 and stochastic:
            X = X * T_rng.binomial(self.input_shape, n=1, p=self.p, dtype=floatX)
        return X


class Dropconnect(DenseLayer):
    def __init__(self, incoming, nb_units, corruption_level=0.5, name=None,
                 W=glorot_uniform, b=(constant, {'value':0.0}),
                 activation=tanh):
        super(Dropconnect, self).__init__(incoming, nb_units, name=name,
                 W=W, b=b, activation=activation)
        self.p = 1 - corruption_level

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.p > 0 and stochastic:
            self.W = self.W * T_rng.binomial(self.shape, n=1, p=self.p, dtype=floatX)
        return self.activation(T.dot(X, self.W) + self.b)
