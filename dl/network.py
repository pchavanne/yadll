# -*- coding: UTF-8 -*-

from .hyperparameters import *
from .model import *

__all__ = ['logistic_regression',
           'mlp',
           'dropout',
           'dropconnect',
           'autoencoder',
           'denoising_autoencoder',
           'stacked_denoising_autoencoder',
           'rbm',
           'dbn',
           'lstm'
           ]


def logistic_regression(input_var=None, shape=None):
    # Create connected layers
    l_in = InputLayer(shape=shape, input_var=input_var, name='Input')
    l_out = LogisticRegression(incoming=l_in, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = Network('logistic_regression')
    net.add(l_in)
    net.add(l_out)
    return net


def mlp(input_var=None, shape=None):
    # Create connected layers
    l_in = InputLayer(shape=shape, input_var=input_var, name='Input')
    l_hid1 = DenseLayer(incoming=l_in, nb_units=500, W=glorot_uniform,
                        activation=relu, name='Hidden layer 1')
    l_hid2 = DenseLayer(incoming=l_hid1, nb_units=500, W=glorot_uniform,
                        activation=relu, name='Hidden layer 2')
    l_out = LogisticRegression(incoming=l_hid2, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = Network('mlp')
    net.add(l_in)
    net.add(l_hid1)
    net.add(l_hid2)
    net.add(l_out)
    return net


def dropout(input_var=None, shape=None):
    # Create connected layers
    l_in = InputLayer(shape=shape, input_var=input_var, name='Input')
    l_dro1 = Dropout(incoming=l_in, corruption_level=0.4, name='Dropout 1')
    l_hid1 = DenseLayer(incoming=l_dro1, nb_units=500, W=glorot_uniform,
                        activation=relu, name='Hidden layer 1')
    l_dro2 = Dropout(incoming=l_hid1, corruption_level=0.2, name='Dropout 2')
    l_hid2 = DenseLayer(incoming=l_dro2, nb_units=500, W=glorot_uniform,
                        activation=relu, name='Hidden layer 2')
    l_out = LogisticRegression(incoming=l_hid2, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = Network('dropout')
    net.add(l_in)
    net.add(l_dro1)
    net.add(l_hid1)
    net.add(l_dro2)
    net.add(l_hid2)
    net.add(l_out)
    return net


def dropconnect(input_var=None, shape=None):
    # Create connected layers
    l_in = InputLayer(shape=shape, input_var=input_var, name='Input')
    l_dc1 = Dropconnect(incoming=l_in, nb_units=500, corruption_level=0.4,
                        W=glorot_uniform, activation=relu, name='Dropconnect layer 1')
    l_dc2 = Dropconnect(incoming=l_dc1, nb_units=500, corruption_level=0.2,
                        W=glorot_uniform, activation=relu, name='Dropconnect layer 2')
    l_out = LogisticRegression(incoming=l_dc2, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = Network('dropconnect')
    net.add(l_in)
    net.add(l_dc1)
    net.add(l_dc2)
    net.add(l_out)
    return net


def autoencoder(input_var=None, shape=None):
    # Unsupervised hyperparameters
    hp_ae = Hyperparameters()
    hp_ae('batch_size', 10)
    hp_ae('n_epochs', 15)
    hp_ae('learning_rate', 0.01)
    # Create connected layers
    l_in = InputLayer(shape=shape, input_var=input_var, name='Input')
    l_ae1 = AutoEncoder(incoming=l_in, nb_units=100, hyperparameters=hp_ae,
                        corruption_level=0.0, name='AutoEncoder')
    l_out = LogisticRegression(incoming=l_ae1, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = Network('autoencoder')
    net.add(l_in)
    net.add(l_ae1)
    net.add(l_out)
    return net


def denoising_autoencoder(input_var=None, shape=None):
    # Unsupervised hyperparameters
    hp_ae = Hyperparameters()
    hp_ae('batch_size', 10)
    hp_ae('n_epochs', 15)
    hp_ae('learning_rate', 0.01)
    # Create connected layers
    l_in = InputLayer(shape=shape, input_var=input_var, name='Input')
    l_ae1 = AutoEncoder(incoming=l_in, nb_units=500, hyperparameters=hp_ae,
                        corruption_level=0.3, name='Denoising AutoEncoder')
    l_out = LogisticRegression(incoming=l_ae1, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = Network('denoising_autoencoder')
    net.add(l_in)
    net.add(l_ae1)
    net.add(l_out)
    return net


def stacked_denoising_autoencoder(input_var=None, shape=None):
    # Unsupervised hyperparameters
    hp_ae = Hyperparameters()
    hp_ae('batch_size', 10)
    hp_ae('n_epochs', 15)
    hp_ae('learning_rate', 0.01)
    # Create connected layers
    l_in = InputLayer(shape=shape, input_var=input_var, name='Input')
    l_ae1 = AutoEncoder(incoming=l_in, nb_units=500, hyperparameters=hp_ae,
                        corruption_level=0.2, name='Denoising AutoEncoder 1')
    l_ae2 = AutoEncoder(incoming=l_ae1, nb_units=500, hyperparameters=hp_ae,
                        corruption_level=0.4, name='Denoising AutoEncoder 2')
    l_out = LogisticRegression(incoming=l_ae2, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = Network('stacked_denoising_autoencoder')
    net.add(l_in)
    net.add(l_ae1)
    net.add(l_ae2)
    net.add(l_out)
    return net


def rbm(input_var=None, shape=None):
    # Unsupervised hyperparameters
    hp_ae = Hyperparameters()
    hp_ae('batch_size', 10)
    hp_ae('n_epochs', 15)
    hp_ae('learning_rate', 0.01)
    # Create connected layers
    l_in = InputLayer(shape=shape, input_var=input_var, name='Input')
    l_rbm1 = RBM(incoming=l_in, nb_units=500, hyperparameters=hp_ae,
                 name='Restricted Boltzmann Machine')
    l_out = LogisticRegression(incoming=l_rbm1, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = Network('rbm')
    net.add(l_in)
    net.add(l_rbm1)
    net.add(l_out)
    return net


def dbn(input_var=None, shape=None):
    # Unsupervised hyperparameters
    hp_ae = Hyperparameters()
    hp_ae('batch_size', 10)
    hp_ae('n_epochs', 15)
    hp_ae('learning_rate', 0.01)
    # Create connected layers
    l_in = InputLayer(shape=shape, input_var=input_var, name='Input')
    l_rbm1 = RBM(incoming=l_in, nb_units=500, hyperparameters=hp_ae,
                 name='Restricted Boltzmann Machine 1')
    l_rbm2 = RBM(incoming=l_in, nb_units=500, hyperparameters=hp_ae,
                 name='Restricted Boltzmann Machine 2')
    l_out = LogisticRegression(incoming=l_rbm2, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = Network('dbn')
    net.add(l_in)
    net.add(l_rbm1)
    net.add(l_out)
    return net


def lstm(input_var=None, shape=None):
    # Create connected layers
    l_in = InputLayer(shape=shape, input_var=input_var, name='Input')
    l_lstm = RBM(incoming=l_in, nb_units=500, hyperparameters=hp_ae,
                 name='Long Short Term Memory')
    l_out = LogisticRegression(incoming=l_lstm, nb_class=10, name='Logistic regression')
    # Create network and add layers
    net = Network('lstm')
    net.add(l_in)
    net.add(l_lstm)
    net.add(l_out)
    return net

