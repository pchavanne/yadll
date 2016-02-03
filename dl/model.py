# -*- coding: UTF-8 -*-

import cPickle

from .layers import *


class Network(object):
    def __init__(self, name=None, layers=None):
        if not layers:
            self.layers = []
        else:
            for layer in layers:
                self.add(layer)
        self.params = []
        self.reguls = 0
        self.has_unspervised_layer = False
        self.name = name

    def add(self, layer):
        self.layers.append(layer)
        self.params.extend(layer.params)
        self.reguls += layer.reguls
        if isinstance(layer, UnsupervisedLayer):
            self.has_unspervised_layer = True

    def params(self):
        return self.params

    def reguls(self):
        return self.reguls

    def get_output(self, **kwargs):
        return self.layers[-1].get_output(**kwargs)


class Model(object):
    def __init__(self, network=None, data=None, hyperparameters=None, name=None,
                 updates=sgd_updates, file=None):
        self.network = network
        self.data = data             # data [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
        self.name = name
        self.hp = hyperparameters
        self.updates = updates
        self.file = file
        self.index = T.iscalar()     # index to a [mini]batch
        self.x = T.matrix('x')       # the input data is presented as a matrix
        if data.train_set_y.ndim == 1:
            self.y = T.ivector('y')      # the output labels are presented as 1D vector of[int] labels
        else:
            self.y = T.matrix('y')

    @timer(' Unsupervised Pre-Training')
    def pretrain(self):
        for layer in self.network.layers:
            if isinstance(layer, UnsupervisedLayer):
                layer.unsupervised_training(self.x, self.data.train_set_x)

    @timer(' Training')
    def train(self, unsupervised_training=True, save_model=False):

        if unsupervised_training and self.network.has_unspervised_layer:
            self.pretrain()

        n_train_batches = self.data.train_set_x.get_value(borrow=True).shape[0] / self.hp.batch_size
        n_valid_batches = self.data.valid_set_x.get_value(borrow=True).shape[0] / self.hp.batch_size
        n_test_batches = self.data.test_set_x.get_value(borrow=True).shape[0] / self.hp.batch_size

        cost = -T.mean(T.log(self.network.get_output(stochastic=True))[T.arange(self.y.shape[0]), self.y])
        # add regularistion
        cost += self.network.reguls

        # updates of the model as a list of (variable, update expression) pairs
        updates = self.updates(cost, self.network.params, self.hp.learning_rate)

        # compiling Theano functions for training, validating and testing the model
        train_model = theano.function(inputs=[self.index], outputs=cost, updates=updates, name='train',
                                      givens={self.x: self.data.train_set_x[self.index * self.hp.batch_size: (self.index + 1) * self.hp.batch_size],
                                              self.y: self.data.train_set_y[self.index * self.hp.batch_size: (self.index + 1) * self.hp.batch_size]})

        ################################################
        # Validation & Test functions
        prediction = T.argmax(self.network.get_output(stochastic=False), axis=1)
        error = T.neq(prediction, self.y)

        validate_model = theano.function(inputs=[self.index], outputs=error, name='validate',
                                         givens={self.x: self.data.valid_set_x[self.index * self.hp.batch_size:(self.index + 1) * self.hp.batch_size],
                                                 self.y: self.data.valid_set_y[self.index * self.hp.batch_size:(self.index + 1) * self.hp.batch_size]})

        test_model = theano.function(inputs=[self.index], outputs=error, name='test',
                                     givens={self.x: self.data.test_set_x[self.index * self.hp.batch_size:(self.index + 1) * self.hp.batch_size],
                                             self.y: self.data.test_set_y[self.index * self.hp.batch_size:(self.index + 1) * self.hp.batch_size]})

        print '... Training the model'

        # early-stopping parameters
        patience = self.hp.patience  # look at this many batches regardless
        patience_increase = 2  # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is considered significant
        validation_frequency = min(n_train_batches, patience / 2)  # go through this many minibatche before checking the network

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        epoch = 0
        done_looping = False

        while (epoch < self.hp.n_epochs) and (not done_looping):
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

                    print('epoch %i, minibatch %i/%i, validation error %.3f %%' %
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

                        print('  epoch %i, minibatch %i/%i, test error of best model %.3f %%' %
                              (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

                        # # save the best model
                        # if save_model:
                        #
                        #     with open(self.file, 'wb') as f:
                        #         cPickle.dump(self.network, f)

                if patience <= iter:
                    done_looping = True
                    break

        print '\n Optimization completed. ' + ('Early stopped at epoch: %i' % epoch) \
            if done_looping else 'Optimization completed. ' + ('Trained on all %i epochs' % epoch)

        print(' Validation score of %.3f %% obtained at iteration %i, with test performance %.3f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))

        # if save_model:
        #     print ' Model saved as: ' + network.file if save_model else ' Model not saved!!'

