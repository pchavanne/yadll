# -*- coding: UTF-8 -*-


import numpy as np

import theano
import theano.tensor as T

import itertools
from collections import OrderedDict

float_X = theano.config.floatX


def floatX(arr):
    return np.asarray(arr, dtype=float_X)


class Hyperparameters(object):
    def __init__(self):
        self.hp_value = OrderedDict()
        self.hp_default = OrderedDict()
        self.hp_range = OrderedDict()
        self.iteration = 0

    def __call__(self, name, value=None, range=None):
        self.__setattr__(name, value)
        self.hp_value[name] = value
        self.hp_default[name] = value
        self.hp_range[name] = range
        if not range:
            self.hp_range[name] = [value]
        product = [x for x in apply(itertools.product, self.hp_range.values())]
        self.hp_product = [dict(zip(self.hp_value.keys(), p)) for p in product]

    def __str__(self):
        return str(self.hp_value)

    def __iter__(self):
        return self

    def next(self):
        if self.iteration > len(self.hp_product) - 1:
            raise StopIteration
        self.hp_value = self.hp_product[self.iteration]
        for name, value in self.hp_value.iteritems():
            self.__setattr__(name, value)
        self.iteration += 1
        return self

    def reset(self):
        self.hp_value = self.hp_default
        self.iteration = 0


def collect_shared_vars(expressions):
    # wrap single expression in list
    if isinstance(expressions, theano.Variable):
        expressions = [expressions]
    # return list of all shared variables
    return [v for v in theano.gof.graph.inputs(reversed(expressions))
            if isinstance(v, theano.compile.SharedVariable)]


def create_param(spec, shape, name=None):
    shape = tuple(shape)  # convert to tuple if needed
    if any(d <= 0 for d in shape):
        raise ValueError((
            "Cannot create param with a non-positive shape dimension. "
            "Tried to create param with shape=%r, name=%r") % (shape, name))

    if isinstance(spec, theano.Variable):
        if spec.ndim != len(shape):
            raise RuntimeError("parameter variable has %d dimensions, "
                               "should be %d" % (spec.ndim, len(shape)))
        return spec

    elif isinstance(spec, np.ndarray):
        if spec.shape != shape:
            raise RuntimeError("parameter array has shape %s, should be "
                               "%s" % (spec.shape, shape))
        return theano.shared(spec, name=name)

    elif hasattr(spec, '__call__'):
        arr = spec(shape)
        try:
            arr = floatX(arr)
        except Exception:
            raise RuntimeError("cannot initialize parameters: the "
                               "provided callable did not return an "
                               "array-like value")
        if arr.shape != shape:
            raise RuntimeError("cannot initialize parameters: the "
                               "provided callable did not return a value "
                               "with the correct shape")
        return theano.shared(arr, name=name)

    else:
        raise RuntimeError("cannot initialize parameters: 'spec' is not "
                           "a numpy array, a Theano expression, or a "
                           "callable")