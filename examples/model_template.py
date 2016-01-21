#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import timeit
import cPickle
import gzip

import numpy as np

import theano
import theano.tensor as T

import dl


def load_data(dataset):
    """
    load_data function should return [train_set, valid_set, test_set]
    """
    print '... Loading the data'

    start_time = timeit.default_timer()

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

    end_time = timeit.default_timer()

    print ' Loading the data took %.2f s' % ((end_time - start_time) / 60.)

    return rval


def build_network(input_var=None):
    l_in = dl.layers.Input_Layer(shape=(None, 28 * 28), input_var=input_var)
    l_out = dl.layers.Dense_Layer(incoming=l_in, nb_units=800,
                                  activation=dl.activation.softmax)
    return l_out


def train(hp, dataset, save_model=False):
################################################
##              LOAD DATA                     ##
################################################

    dataset = dataset

    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / hp.batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / hp.batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / hp.batch_size

################################################
##             BUILD MODEL                    ##
################################################

    print '... Building the model'

    start_time = timeit.default_timer()

    # allocate symbolic variables for the data
    noise = T.bscalar()     # noise set to 1 when training with noise, 0 for validate and test
    index = T.lscalar()     # index to a [mini]batch
    x = T.matrix('x')       # the input data is presented as a matrix
    y = T.ivector('y')      # the output labels are presented as 1D vector of[int] labels


    # construct the network
    network = build_network(input_var=x)

    # the cost we minimize during training
    cost = network.get_output()

    # compute the gradient of cost with respect to params
    gparams = [T.grad(cost, param) for param in network.params]

    # specify how to update the parameters of the model as a list of (variable, update expression) pairs
    updates = dl.updates.sgd_updates(gparams, network.params, hp.learning_rate)

    # compiling Theano functions for training, validating and testing the model
    train_model = theano.function(inputs=[index, theano.Param(noise, default=1)],
                                  outputs=cost, updates=updates, name='train',
                                  givens={x: train_set_x[index * hp.batch_size: (index + 1) * hp.batch_size],
                                          y: train_set_y[index * hp.batch_size: (index + 1) * hp.batch_size]})

    validate_model = theano.function(inputs=[index, theano.Param(noise, default=0)],
                                     outputs=network.errors(y), name='validate',
                                     givens={x: valid_set_x[index * hp.batch_size:(index + 1) * hp.batch_size],
                                             y: valid_set_y[index * hp.batch_size:(index + 1) * hp.batch_size]})

    test_model = theano.function(inputs=[index, theano.Param(noise, default=0)],
                                 outputs=network.errors(y), name='test',
                                 givens={x: test_set_x[index * hp.batch_size:(index + 1) * hp.batch_size],
                                         y: test_set_y[index * hp.batch_size:(index + 1) * hp.batch_size]})

    end_time = timeit.default_timer()

    print ' Building the model took %.2f s' % ((end_time - start_time) / 60.)

################################################
##         TRAINING THE MODEL                 ##
################################################

    print '... Training the model'

    start_time = timeit.default_timer()

    # early-stopping parameters
    patience = 10000  # look at this many batches regardless
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

    print str(network)

    print hp

    print ' Optimization completed. ' + ('Early stopped at epoch: %i' % epoch) \
        if done_looping else 'Optimization completed. ' + ('Trained on all %i epochs' % epoch)
    print(' Validation score of %f %% obtained at iteration %i, with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    s = end_time - start_time
    print ' Training the model took %d h %02d m %02d s' % (s / 3600, s / 60 % 60, s % 60)

    print ' Model saved as: ' + network.file if save_model else ' Model not saved!!'

    report = OrderedDict()
    report['index'] = hp.iteration
    report['network'] = str(network)
    report['file'] = network.file
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
    # Load dataset
    datafile = '/home/philippe/Python/Theano/mnist.pkl.gz'
    dataset = load_data(datafile)

    # Load Hyperparameters
    hp = Hyperparameters()
    hp('batch_size', 20, [5, 10, 15, 20])
    hp('n_epochs', 50)
    hp('learning_rate', 0.01, [0.001, 0.01, 0.1, 1])
    hp('l1_reg', 0.00, [0.0001, 0.001])
    hp('l2_reg', 0.000)

    grid_search = False

    # train model or find hyperparameters
    reports = []
    if grid_search:
        for _ in hp:
            reports.append(train(hp=hp, dataset=dataset, save_model=False))
    else:
        reports.append(train(hp=hp, dataset=dataset, save_model=False))

    reports = pd.DataFrame(reports)
    param_reports = pd.DataFrame.from_records(reports['parameters'])
    pd_report = pd.DataFrame(reports,
                             columns=['iteration', 'test', 'validation', 'training time'])
    reports = pd.concat([param_reports, pd_report], axis=1)

    reports.to_html(open('/home/philippe/Python/Theano/dl/report.html', 'w'))

    print reports.loc[reports['validation'].idxmin()]
