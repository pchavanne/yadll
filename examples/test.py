#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
"""
import logging

import numpy as np
import yadll
import examples.networks as networks

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

_networks = [ 'logistic_regression',
            'mlp',
            'dropout',
            'dropconnect',
            'convpool',
            'lenet5',
            'autoencoder',
            'denoising_autoencoder',
            'gaussian_denoising_autoencoder',
            'contractive_denoising_autoencoder',
            'stacked_denoising_autoencoder',
            'rbm',
            'dbn',
            'batch_normalization',
           ]

# load the data
data = yadll.data.Data(yadll.data.mnist_loader())


def build_network(network='Logistic_regression', input_var=None):
    network_builder = getattr(networks, network)
    return network_builder(input_var=input_var)


for network in _networks:
    ################################################
    # construct the model
    model = yadll.model.Model(name=network, data=data, file='best_model.ym')
    # construct the network
    net, hp = build_network(network)
    # add the network to the model
    model.network = net
    # add the hyperparameters to the model
    model.hp = hp
    # updates method
    model.updates = yadll.updates.momentum
    # Compile model
    model.compile('all')
    # Saving configuration of the model. Model doesn't have to be trained
    conf = model.to_conf()    # get the configuration
    model.to_conf('conf.yc')  # or save it to file .yc by convention
    # train the model
    model.train(unsupervised_training=True)
    # Saving network parameters after training
    net.save_params('net_params.yp')

    # We can test it on some examples from test
    test_set_x = data.test_set_x.get_value()
    test_set_y = data.test_set_y.get_value()

    predicted_values = [np.argmax(prediction) for prediction in model.predict(test_set_x[:30])]
    true_values = [np.argmax(true_value) for true_value in test_set_y[:30]]

    print ("Model 1 Predicted & True values for the first 30 examples in test set:")
    print(predicted_values)
    print(true_values)

    ##########################################################################
    # Loading model from file
    model_2 = yadll.model.load_model('best_model.ym')
    # model is ready to use we can make prediction directly.
    # Watch out this not the proper way of saving models.
    predicted_values_2 = [np.argmax(prediction) for prediction in model_2.predict(test_set_x[:30])]

    print ("Model 2 Predicted & True values for the first 30 examples in test set:")
    print(predicted_values_2)
    print(true_values)
    ##########################################################################
    # Recreate model and load parameters
    model_3 = yadll.model.Model(data=data)
    model_3.network = build_network(network)[0]
    # Network as been re-created so parameters has just been initialized
    # Let's try prediction with this network.
    predicted_values_3 = [np.argmax(prediction) for prediction in model_3.predict(test_set_x[:30])]
    print ("Model 3 without loading parameters values for the first 30 examples in test set:")
    print(predicted_values_3)
    print(true_values)
    # Now let's load parameters
    model_3.network.load_params('net_params.yp')
    # And try predicting again
    predicted_values_3 = [np.argmax(prediction) for prediction in model_3.predict(test_set_x[:30])]
    print ("Model 3 after loading parameters values for the first 30 examples in test set:")
    print(predicted_values_3)
    print(true_values)

    ##########################################################################
    # Reconstruction the model from configuration and load parameters
    model_4 = yadll.model.Model()
    model_4.from_conf(conf)         # load from conf obj
    model_5 = yadll.model.Model()
    model_5.from_conf('conf.yc')    # load from conf file

    model_4.network.load_params('net_params.yp')
    model_5.network.load_params('net_params.yp')

    predicted_values_4 = [np.argmax(prediction) for prediction in model_4.predict(test_set_x[:30])]
    print ("Model 4 after loading parameters values for the first 30 examples in test set:")
    print(predicted_values_4)
    print(true_values)

    predicted_values_5 = [np.argmax(prediction) for prediction in model_5.predict(test_set_x[:30])]
    print ("Model 5 after loading parameters values for the first 30 examples in test set:")
    print(predicted_values_5)
    print(true_values)



