# -*- coding: UTF-8 -*-
from .layers import *


class Network(object):
    def __init__(self, name=None, layers=None):
        self.layers = []
        self.params = []
        self.reguls = 0
        self.has_unsupervised_layer = False
        self.name = name
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self.layers.append(layer)
        self.params.extend(layer.params)
        self.reguls += layer.reguls
        if isinstance(layer, UnsupervisedLayer):
            self.has_unsupervised_layer = True

    def params(self):
        return self.params

    def reguls(self):
        return self.reguls

    def get_output(self, **kwargs):
        return self.layers[-1].get_output(**kwargs)
