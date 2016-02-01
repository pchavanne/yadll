#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Example of dl usage on the mnist dataset

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
import timeit
import cPickle
from docopt import docopt

import numpy as np
import pandas as pd

import theano
import theano.tensor as T

import dl

from collections import OrderedDict


@dl.utils.timer(' Loading the data')
def load_data(dataset):
    """
    load_data function should return [train_set, valid_set, test_set]
    """
    print '... Loading the data'
    data = dl.data.Data(dataset)
    return data


def build_network(network_name, input_var=None, shape=None):
    network_builder = getattr(dl.network, network_name)
    return network_builder(input_var=input_var, shape=shape)


def train(network_name, hp, data, save_model=False):
################################################
##              LOAD DATA                     ##
################################################

    data = data

    #train_set_x, train_set_y = dataset[0]
    #valid_set_x, valid_set_y = dataset[1]
    #test_set_x, test_set_y = dataset[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = data.train_set_x.get_value(borrow=True).shape[0] / hp.batch_size
    n_valid_batches = data.valid_set_x.get_value(borrow=True).shape[0] / hp.batch_size
    n_test_batches = data.test_set_x.get_value(borrow=True).shape[0] / hp.batch_size

################################################
##             BUILD MODEL                    ##
################################################

    print '... Building the model'

    start_time = timeit.default_timer()

    ################################################
    # allocate symbolic variables for the data
    index = T.lscalar()     # index to a [mini]batch
    x = T.matrix('x')       # the input data is presented as a matrix
    y = T.ivector('y')      # the output labels are presented as 1D vector of[int] labels

    ################################################
    # construct the network
    network = build_network(network_name, input_var=x, shape=(hp.batch_size, 28 * 28))
    model = dl.model.Model(network, data)

    ################################################
    # Train function

    # the cost we minimize during training
    # cost = dl.objectives.categorical_crossentropy(
    #         prediction=network.get_output(),
    #         target=T.extra_ops.to_one_hot(y, nb_class=10))
    # cost functions
    cost = -T.mean(T.log(network.get_output(stochastic=True))[T.arange(y.shape[0]), y])
    # add regularistion
    cost += network.reguls

    # updates of the model as a list of (variable, update expression) pairs
    updates = dl.updates.sgd_updates(cost, network.params, hp.learning_rate)

    # compiling Theano functions for training, validating and testing the model
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates, name='train',
                                  givens={x: data.train_set_x[index * hp.batch_size: (index + 1) * hp.batch_size],
                                          y: data.train_set_y[index * hp.batch_size: (index + 1) * hp.batch_size]})

    ################################################
    # Validation & Test functions
    prediction = T.argmax(network.get_output(stochastic=False), axis=1)
    error = T.neq(prediction, y)

    validate_model = theano.function(inputs=[index], outputs=error, name='validate',
                                     givens={x: data.valid_set_x[index * hp.batch_size:(index + 1) * hp.batch_size],
                                             y: data.valid_set_y[index * hp.batch_size:(index + 1) * hp.batch_size]})

    test_model = theano.function(inputs=[index], outputs=error, name='test',
                                 givens={x: data.test_set_x[index * hp.batch_size:(index + 1) * hp.batch_size],
                                         y: data.test_set_y[index * hp.batch_size:(index + 1) * hp.batch_size]})

    end_time = timeit.default_timer()

    print ' Building the model took %.2f s' % ((end_time - start_time) / 60.)


################################################
##   PRETRAINING UNSUPERVISED LAYERS          ##
################################################
    # for layer in network.layers:
    #     if isinstance(layer, dl.layers.UnsupervisedLayer):
    #         start_time = timeit.default_timer()
    #         print '... Pretraining the layer: %s' % layer.name
    #         n_train_batches = data.train_set_x.get_value(borrow=True).shape[0] / layer.hp.batch_size
    #         cost = layer.get_unsupervised_cost(stochastic=True)
    #         updates = dl.updates.sgd_updates(cost, layer.unsupervised_params, layer.hp.learning_rate)
    #         pretrain = theano.function(inputs=[index], outputs=cost, updates=updates,
    #                                    givens={x: data.train_set_x[index * layer.hp.batch_size: (index + 1) * layer.hp.batch_size]})
    #         for epoch in xrange(layer.hp.n_epochs):
    #             c = []
    #             for minibatch_index in xrange(n_train_batches):
    #                 c.append(pretrain(minibatch_index))
    #             print 'Layer: %s, pretraining epoch %d, cost %d' % (layer.name, epoch, np.mean(c))
    #         end_time = timeit.default_timer()
    #         s = end_time - start_time
    #         print ' Pretraining the layer %s took %d h %02d m %02d s' % (layer.name, s / 3600, s / 60 % 60, s % 60)
    model.pretrain()
################################################
##         TRAINING THE MODEL                 ##
################################################

    print '... Training the model'

    start_time = timeit.default_timer()

    # early-stopping parameters
    patience = 5000  # look at this many batches regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2) # go through this many minibatche before checking the network

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    epoch = 0
    done_looping = False

    while (epoch < hp.n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            # train
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print('  epoch %i, minibatch %i/%i, test error of best model %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

                    # save the best model
                    if save_model:
                        with open(network.file, 'wb') as f:
                            cPickle.dump(network, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    print '\nOptimization completed. ' + ('Early stopped at epoch: %i' % epoch) \
        if done_looping else 'Optimization completed. ' + ('Trained on all %i epochs' % epoch)

    print('\n Validation score of %f %% obtained at iteration %i, with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    s = end_time - start_time
    print ' Training the model took %d h %02d m %02d s' % (s / 3600, s / 60 % 60, s % 60)

    print ' Model saved as: ' + network.file if save_model else ' Model not saved!!'

    report = OrderedDict()
    report['index'] = hp.iteration
#    report['file'] = network.file
    report['parameters'] = hp.hp_value
    report['iteration'] = best_iter
    report['validation'] = best_validation_loss * 100.
    report['test'] = test_score * 100.
    report['training time'] = '%d h %02d m %02d s' % (s / 3600, s / 60 % 60, s % 60)

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
        for d in dl.network.__all__:
            print '\t%s' % d

    else:
        if arguments['<network>']:
            network_name = arguments['<network>']
        if network_name not in dl.network.__all__:
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

        reports = pd.DataFrame(reports)
        param_reports = pd.DataFrame.from_records(reports['parameters'])
        pd_report = pd.DataFrame(reports,
                                 columns=['iteration', 'test', 'validation', 'training time'])
        reports = pd.concat([param_reports, pd_report], axis=1)

        reports.to_html(open('/home/philippe/Python/dl/report.html', 'w'))

        print reports.loc[reports['validation'].idxmin()]
