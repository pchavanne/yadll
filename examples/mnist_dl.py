#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Example of dl usage on the mnist dataset
use -n or --network_list to see all available networks

Usage:
    mnist_dl.py [<network>] [default:'logistic_regression']
    mnist_dl.py (-n | --network_list)
    mnist_dl.py (-h | --help)
    mnist_dl.py --version

Options:
    -n --network_list   Show available networks
    -h --help           Show this screen
    --version           Show version
"""
import cPickle

import theano
from docopt import docopt

import dl
import examples.network as network


@dl.utils.timer(' Loading the data')
def load_data(dataset):
    print '... Loading the data'
    return dl.data.Data(dataset)


def build_network(network_name='Logistic_regression', input_var=None):
    network_builder = getattr(network, network_name)
    return network_builder(input_var=input_var)


def train(network_name, data):

    ################################################
    # construct the model
    model = dl.model.Model(name=network_name, data=data)
    # construct the network
    network , hp = build_network(network_name, input_var=model.x)
    # add the network to the model
    model.network = network
    # add the hyperparameters to the model
    model.hp = hp
    # updates method
    model.updates = dl.updates.sgd_updates
    # train the model
    model.train(unsupervised_training=True)



def predict(dataset):
    # load the saved model
    model = cPickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(inputs=[model.input], outputs=model.output)

    # We can test it on some examples from test test
    test_set_x, test_set_y = dataset[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print ("Predicted values for the first 10 examples in test set:")
    print predicted_values


if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.0.1')
    network_name = 'logistic_regression'
    if arguments['--network_list']:
        print 'Default network is: %s' % network_name
        print 'Supported networks are:'
        for d in network.__all__:
            print '\t%s' % d

    else:
        if arguments['<network>']:
            network_name = arguments['<network>']
        if network_name not in network.__all__:
            raise TypeError('netwok name provided is not supported. Check supported network'
                            ' with option -n')
        # Load dataset
        datafile = '/home/philippe/Python/Theano/mnist.pkl.gz'
        data = load_data(datafile)

        train(network_name, data=data)

