# -*- coding: UTF-8 -*-
from collections import OrderedDict
import cPickle
import yadll
from .layers import *

import logging

logger = logging.getLogger(__name__)


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
        self.layer_names = []
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
        layer : :class: `Layer`

        """
        self.layers.append(layer)
        self.layer_names.append(layer.name)
        self.params.extend(layer.params)
        self.reguls += layer.reguls
        if isinstance(layer, UnsupervisedLayer):
            self.has_unsupervised_layer = True

    def get_output(self, **kwargs):
        """
        Returns the output of the network

        Returns
        -------
        symbolic expresssion
            output of the network

        """
        return self.layers[-1].get_output(**kwargs)

    def get_layer(self, layer_name):
        """
        Get a layer of the network from its name

        Parameters
        ----------
        layer_name : `string`
            name of the layer requested

        Returns
        -------
        yaddl.layers object
            a yadll layer object in the

        """
        return self.layers[self.layer_names.index(layer_name)]

    def __getitem__(self, layer_name):
        return self.get_layer(layer_name)

    def save_params(self, file):
        """
        Save the parameters of the network to file with cPickle

        Parameters
        ----------
        file : `string`
            file name

        Examples
        --------

        >>> my_network.save_params('my_network_params.yp')

        """
        if self.name is None:
            logger.error(
                'Your network has no name. Please set one and try again.')
            return

        with open(file, 'wb') as f:
            cPickle.dump(self.name, f, cPickle.HIGHEST_PROTOCOL)
            for param in self.params:
                cPickle.dump(param.get_value(borrow=True), f,
                             cPickle.HIGHEST_PROTOCOL)

    def load_params(self, file):
        """
        load (unpickle) saved parameters of a network.

        Parameters
        ----------
        file : `string'
            name of the file containing the saved parameters


        Examples
        --------

        >>> my_network.load_params('my_network_params.yp')

        """

        with open(file, 'rb') as f:
            pickled_name = cPickle.load(f)
            if pickled_name != self.name:
                logger.error(
                    'Network names are different. Saved network name is: %s' % pickled_name)
                return

            for param in self.params:
                param.set_value(cPickle.load(f), borrow=True)

    def to_conf(self):
        return {'name': self.name,
                'has_unsupervised_layer': self.has_unsupervised_layer,
                'layers': OrderedDict([(layer.name, layer.to_conf()) for layer in self.layers])}

    def from_conf(self, conf):
        layers = conf.pop('layers', None)
        self.__dict__.update(conf)
        for l in layers.items():
            layer_class = getattr(yadll.layers, l[1]['type'])
            if layer_class is yadll.layers.InputLayer:
                layer = layer_class(**l[1])
            else:
                layer = layer_class(**dict({'incoming': self.layers[-1]}, **l[1])) # ony work for sequential nets
            self.add(layer)
