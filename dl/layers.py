# -*- coding: UTF-8 -*-

from .init import *
from .utils import *
from .objectives import *

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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
        self.reguls = 0

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
                 activation=tanh, l1=None, l2=None):
        super(DenseLayer, self).__init__(incoming, name)
        self.shape = (self.input_shape[1], nb_units)
        if isinstance(W, theano.compile.SharedVariable):
            self.W = W
        else:
            self.W = initializer(W, shape=self.shape, name='W')
        self.params.append(self.W)
        if isinstance(b, theano.compile.SharedVariable):
            self.b = b
        else:
            self.b = initializer(b, shape=(self.shape[1],), name='b')
        self.params.append(self.b)
        self.activation = activation
        if l1:
            self.reguls += l1 * T.mean(T.abs_(self.W))
        if l2:
            self.reguls += l2 * T.mean(T.sqr(self.W))

    @property
    def output_shape(self):
        return self.input_shape[0], self.shape[1]

    def get_reguls(self):
        return self.reguls

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
                 activation=tanh, **kwargs):
        super(Dropconnect, self).__init__(incoming, nb_units, name=name,
                 W=W, b=b, activation=activation, **kwargs)
        self.p = 1 - corruption_level

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.p > 0 and stochastic:
            self.W = self.W * T_rng.binomial(self.shape, n=1, p=self.p, dtype=floatX)
        return self.activation(T.dot(X, self.W) + self.b)


class AutoEncoder(DenseLayer):
    def __init__(self, incoming, nb_units, corruption_level=0.5, name=None,
                 W=glorot_uniform, b=(constant, {'value':0.0}),
                 activation=tanh, **kwargs):
        super(AutoEncoder, self).__init__(incoming, nb_units, name=name,
                                          W=W, b=b, activation=activation, **kwargs)
        self.W_prime = self.W.T
        self.b_prime = initializer(b, shape=(self.shape[1],), name='b_prime')
        self.auto_params = self.params
        self.auto_params.append(self.b_prime)
        self.p = 1 - corruption_level

    def get_encoded_input(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.p > 0 and stochastic:
            X = X * T_rng.binomial(self.input_shape, n=1, p=self.p, dtype=floatX)
        Y = self.activation(T.dot(X, self.W) + self.b)
        Z = self.activation(T.dot(Y, self.W_prime) + self.b_prime)
        return Z

    def get_encoding_cost(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        Z = self.get_encoded_input(stochastic=stochastic, **kwargs)
        cost = T.mean(categorical_crossentropy(Z, X))
        return cost


class RBM(DenseLayer):
    def __init__(self, incoming, nb_units, corruption_level=0.5, name=None,
                 W=glorot_uniform, b=(constant, {'value':0.0}),
                 activation=tanh, **kwargs):
        super(RBM, self).__init__(incoming, nb_units, name=name,
                                          W=W, b=b, activation=activation, **kwargs)
