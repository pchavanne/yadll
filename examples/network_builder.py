#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import dl


def build_logistic_regression(input_var=None, batch_size=None):
    # Create connected layers
    l_in = dl.layers.InputLayer(shape=(batch_size, 28 * 28), input_var=input_var, name='Input')
    l_out = dl.layers.LogisticRegression(incoming=l_in, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = dl.model.Network()
    net.add(l_in)
    net.add(l_out)
    return net


def build_mlp(input_var=None, batch_size=None):
    # Create connected layers
    l_in = dl.layers.InputLayer(shape=(batch_size, 28 * 28), input_var=input_var, name='Input')
    l_hid1 = dl.layers.DenseLayer(incoming=l_in, nb_units=500, W=dl.init.glorot_uniform,
                                  activation=dl.activation.relu, name='Hidden layer 1')
    l_hid2 = dl.layers.DenseLayer(incoming=l_hid1, nb_units=500, W=dl.init.glorot_uniform,
                                  activation=dl.activation.relu, name='Hidden layer 2')
    l_out = dl.layers.LogisticRegression(incoming=l_hid2, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = dl.model.Network()
    net.add(l_in)
    net.add(l_hid1)
    net.add(l_hid2)
    net.add(l_out)
    return net


def build_dropout(input_var=None, batch_size=None):
    # Create connected layers
    l_in = dl.layers.InputLayer(shape=(batch_size, 28 * 28), input_var=input_var, name='Input')
    l_dro1 = dl.layers.Dropout(incoming=l_in, corruption_level=0.4, name='Dropout 1')
    l_hid1 = dl.layers.DenseLayer(incoming=l_dro1, nb_units=500, W=dl.init.glorot_uniform,
                                  activation=dl.activation.relu, name='Hidden layer 1')
    l_dro2 = dl.layers.Dropout(incoming=l_hid1, corruption_level=0.2, name='Dropout 2')
    l_hid2 = dl.layers.DenseLayer(incoming=l_dro2, nb_units=500, W=dl.init.glorot_uniform,
                                  activation=dl.activation.relu, name='Hidden layer 2')
    l_out = dl.layers.LogisticRegression(incoming=l_hid2, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = dl.model.Network()
    net.add(l_in)
    net.add(l_dro1)
    net.add(l_hid1)
    net.add(l_dro2)
    net.add(l_hid2)
    net.add(l_out)
    return net


def build_dropconnect(input_var=None, batch_size=None):
    # Create connected layers
    l_in = dl.layers.InputLayer(shape=(batch_size, 28 * 28), input_var=input_var, name='Input')
    l_dc1 = dl.layers.Dropconnect(incoming=l_in, nb_units=500, corruption_level=0.4,
                                  W=dl.init.glorot_uniform, activation=dl.activation.relu,
                                  name='Hidden layer 1')
    l_dc2 = dl.layers.Dropconnect(incoming=l_dc1, nb_units=500, corruption_level=0.2,
                                  W=dl.init.glorot_uniform, activation=dl.activation.relu,
                                  name='Hidden layer 2')
    l_out = dl.layers.LogisticRegression(incoming=l_dc2, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = dl.model.Network()
    net.add(l_in)
    net.add(l_dc1)
    net.add(l_dc2)
    net.add(l_out)
    return net
