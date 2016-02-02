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
from collections import OrderedDict

import theano
from docopt import docopt

import dl
import examples.network as network


@dl.utils.timer(' Loading the data')
def load_data(dataset):
    print '... Loading the data'
    return dl.data.Data(dataset)


def build_network(network_name='Logistic_regression', input_var=None, shape=None):
    network_builder = getattr(network, network_name)
    return network_builder(input_var=input_var, shape=shape)


def train(network_name, hp, data, save_model=False):

    ################################################
    # construct the model
    model = dl.model.Model(name=network_name, hyperparameters=hp, data=data)
    # construct the network
    network = build_network(network_name, input_var=model.x, shape=(hp.batch_size, 28 * 28))
    # add the network to the model
    model.network = network
    # updates method
    model.updates = dl.updates.sgd_updates
    # train the model
    model.train(unsupervised_training=True)

    report = OrderedDict()
    report['index'] = hp.iteration
# #    report['file'] = network.file
    report['parameters'] = hp.hp_value
#     report['iteration'] = best_iter
#     report['validation'] = best_validation_loss * 100.
#     report['test'] = test_score * 100.
#     report['training time'] = '%d h %02d m %02d s' % (s / 3600, s / 60 % 60, s % 60)

    return report


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

        # Load Hyperparameters
        hp = dl.hyperparameters.Hyperparameters()
        hp('batch_size', 10, [5, 10, 15, 20])
        hp('n_epochs', 1000)
        hp('learning_rate', 0.1, [0.001, 0.01, 0.1, 1])
        hp('l1_reg', 0.00, [0.0001, 0.001])
        hp('l2_reg', 0.000)

        grid_search = False

        # train model or find hyperparameters
        reports = []
        if grid_search:
            for _ in hp:
                reports.append(train(network_name, hp=hp, data=data, save_model=False))
        else:
            reports.append(train(network_name, hp=hp, data=data, save_model=False))

        # reports = pd.DataFrame(reports)
        # param_reports = pd.DataFrame.from_records(reports['parameters'])
        # pd_report = pd.DataFrame(reports,
        #                          columns=['iteration', 'test', 'validation', 'training time'])
        # reports = pd.concat([param_reports, pd_report], axis=1)
        #
        # reports.to_html(open('/home/philippe/Python/dl/report.html', 'w'))
        #
        # print reports.loc[reports['validation'].idxmin()]
