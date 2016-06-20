# -*- coding: UTF-8 -*-
from .layers import *


class Network(object):
    """
    The :class:`Network` class is the container of all the layers of the network.

    Parameters
    ----------
    name : `string`
        The name of the network
    layers : list of :class: `Layer`, optional
        create a network from another network.layers

    Attributes
    ----------
    layers : list of :class: `Layer`
        the list of layers in the network
    params : list of `theano shared variables`
        the list of all the parameters of the network
    reguls : symbolic expression
        regularization cost for the network
    has_unsupervised_layer : `bool`
        True if one of the layer is a subclass of :class: `UnsupervisedLayer`

    """
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
        """
        add a layer to the Network

        Parameters
        ----------
        layer: :class: `Layer`

        """
        self.layers.append(layer)
        self.params.extend(layer.params)
        self.reguls += layer.reguls
        if isinstance(layer, UnsupervisedLayer):
            self.has_unsupervised_layer = True

    def params(self):
        """
        Returns the list of parameters of the network

        Returns
        -------
        params:
            list of parameters of the network

        """
        return self.params

    def reguls(self):
        """
        Returns the regularization cost for the network

        Returns
        -------
        reguls: symbolic expresssion
            regularization cost for the network

        """
        return self.reguls

    def get_output(self, **kwargs):
        """
        Returns the output of the network

        Returns
        -------
        symbolic expresssion
            output of the network

        """
        return self.layers[-1].get_output(**kwargs)
