# -*- coding: UTF-8 -*-
import timeit

class Network(object):
    def __init__(self, layers=None):
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
    def __init__(self, network):
        self.network = network

