# -*- coding: UTF-8 -*-

import itertools


class Hyperparameters(object):
    def __init__(self):
        self.hp_value = dict()
        self.hp_default = dict()
        self.hp_range = dict()
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
        for name, value in self.hp_default.iteritems():
            self.__setattr__(name, value)
        self.iteration = 0

