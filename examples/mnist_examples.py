#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Example of yadll usage on the mnist dataset
many networks are predefined, to see the complete list,
use -n or --network_list to see all available networks

Usage:
    mnist_examples.py [<network>] [default:'logistic_regression']
    mnist_examples.py (-n | --network_list)
    mnist_examples.py (-h | --help)
    mnist_examples.py --version

Options:
    -n --network_list   Show available networks
    -h --help           Show this screen
    --version           Show version
"""
import os
import cPickle
import logging

import theano
from docopt import docopt

import yadll
import examples.networks as networks

logging.basicConfig(level=logging.DEBUG, format='%(message)s')


@yadll.utils.timer(' Loading the data')
def load_data(dataset):
    print '... Loading the data'
    return yadll.data.Data(dataset)


def build_network(network_name='Logistic_regression', input_var=None):
    network_builder = getattr(networks, network_name)
    return network_builder(input_var=input_var)


def train(network_name, data):

    ################################################
    # construct the model
    model = yadll.model.Model(name=network_name, data=data)
    # construct the network
    network, hp = build_network(network_name)
    # add the network to the model
    model.network = network
    # add the hyperparameters to the model
    model.hp = hp
    # updates method
    model.updates = yadll.updates.sgd
    # train the model
    model.train(unsupervised_training=True)

    # We can test it on some examples from test
    test_set_x = data.test_set_x.get_value()
    test_set_y = data.test_set_y.eval()

    predicted_values = model.predict(test_set_x[:30])
    print ("Predicted values for the first 30 examples in test set:")
    print predicted_values
    print test_set_y[:30]


if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.0.1')
    network_name = 'logistic_regression'
    if arguments['--network_list']:
        print 'Default network is: %s' % network_name
        print 'Supported networks are:'
        for d in networks.__all__:
            print '\t%s' % d

    else:
        if arguments['<network>']:
            network_name = arguments['<network>']
        if network_name not in networks.__all__:
            raise TypeError('netwok name provided is not supported. Check supported network'
                            ' with option -n')
        # Load dataset
        datafile = 'mnist.pkl.gz'
        if not os.path.isfile(datafile):
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, datafile)

        data = load_data(datafile)

        train(network_name, data=data)

