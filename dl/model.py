# -*- coding: UTF-8 -*-


class Network(object):
    def __init__(self, layers=[]):
        self.layers = []
        self.params = []
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        self.layers.append(layer)
        self.params.extend(layer.params)

    @property
    def params(self):
        return self.params

    def get_output(self):
        return self.layers[-1].get_output()


class Model(object):
    def __init__(self):
        raise NotImplementedError
