# -*- coding: UTF-8 -*-


class Network(object):
    def __init__(self, layers=None):
        if not layers:
            self.layers = []
        else:
            for layer in layers:
                self.add(layer)
        self.params = []

    def add(self, layer):
        self.layers.append(layer)
        self.params.extend(layer.params)

    def params(self):
        return self.params

    def get_output(self, **kwargs):
        return self.layers[-1].get_output(**kwargs)


class Model(object):
    def __init__(self):
        raise NotImplementedError
