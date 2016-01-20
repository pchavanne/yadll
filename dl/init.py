# -*- coding: UTF-8 -*-

import numpy as np

from .utils import floatX
from .random import get_rng


class Initializer(object):
    def __call__(self, shape):
        return self.sample(shape)

    def sample(self, shape):
        raise NotImplementedError()


class Constant(Initializer):
    def __init__(self, val=0.0):
        self.val = val

    def sample(self, shape):
        return floatX(np.ones(shape) * self.val)


class Normal(Initializer):
    def __init__(self, std=0.01, mean=0.0):
        self.std = std
        self.mean = mean

    def sample(self, shape):
        return floatX(get_rng().normal(self.mean, self.std, size=shape))


class Uniform(Initializer):
    def __init__(self, range=0.01, std=None, mean=0.0):
        if std is not None:
            a = mean - np.sqrt(3) * std
            b = mean + np.sqrt(3) * std
        else:
            try:
                a, b = range  # range is a tuple
            except TypeError:
                a, b = -range, range  # range is a number

        self.range = (a, b)

    def sample(self, shape):
        return floatX(get_rng().uniform(
            low=self.range[0], high=self.range[1], size=shape))


class Glorot(Initializer):
    def __init__(self, initializer, gain=1.0, c01b=False):
        if gain == 'relu':
            gain = np.sqrt(2)

        self.initializer = initializer
        self.gain = gain
        self.c01b = c01b

    def sample(self, shape):
        if self.c01b:
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            n1, n2 = shape[0], shape[3]
            receptive_field_size = shape[1] * shape[2]
        else:
            if len(shape) < 2:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

            n1, n2 = shape[:2]
            receptive_field_size = np.prod(shape[2:])

        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        return self.initializer(std=std).sample(shape)


class GlorotNormal(Glorot):
    def __init__(self, gain=1.0, c01b=False):
        super(GlorotNormal, self).__init__(Normal, gain, c01b)


class GlorotUniform(Glorot):
    def __init__(self, gain=1.0, c01b=False):
        super(GlorotUniform, self).__init__(Uniform, gain, c01b)


