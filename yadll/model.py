# -*- coding: UTF-8 -*-
import cPickle
import yadll
from .layers import *
from .exceptions import *

import logging

logger = logging.getLogger(__name__)


def save_model(model, file=None):
    """
    Save the model to file with cPickle
    This function is used by the training function to save the model.
    Parameters
    ----------
    model : :class:`yadll.model.Model`
        model to be saved in file
    file : `string`
        file name

    """
    if file is None:
        if model.file is None:
            logger.error('No file name. Model not saved.')
            return
        else:
            d_file = model.file
    else:
        d_file = file
    with open(d_file, 'wb') as f:
        cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)


def load_model(file):
    """
    load (unpickle) a saved model

    Parameters
    ----------
    file : `string'
        file name

    Returns
    -------
        a :class:`yadll.model.Model`

    Examples
    --------

    >>> my_model = load_model('my_best_model.ym')

    """
    with open(file, 'rb') as f:
        model = cPickle.load(f)
    return model


class Model(object):
    """
    The :class:`yadll.model.Model` contains the data, the network, the hyperparameters,
    and the report.
    It pre-trains unsupervised layers, trains the network and save it to file.

    Parameters
    ----------
    network : :class:`yadll.network.Network`
        the network to be trained
    data : :class:`yadll.data.Data`
        the training, validating and testing set
    name : `string`
        the name of the model
    updates : :func:`yadll.updates`
        an update function
    file : `string`
        name of the file to save the model. If omitted a name is generated with
        the model name + date + time of training

    """
    def __init__(self, network=None, data=None, hyperparameters=None, name='model',
                 updates=sgd, objective=CCE, evaluation_metric=categorical_accuracy, file=None):
        self.network = network
        self.data = data             # data [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
        self.name = name
        self.hp = hyperparameters
        self.updates = updates
        self.objective = objective
        self.metric = evaluation_metric
        self.file = file
        self.save_mode = None          # None, 'end' or 'each'
        self.index = T.iscalar()       # index to a [mini]batch
        self.epoch_index = T.ivector() # index per epoch
        self.x = T.matrix('x')         # the input data is presented as a matrix
        self.report = dict()

    @timer(' Unsupervised Pre-Training')
    def pretrain(self):
        """
        Pre-training of the unsupervised layers sequentially

        Returns
        -------
            update unsupervised layers weights
        """
        if self.data is None:
            raise NoDataFoundException

        if self.network is None:
            raise NoNetworkFoundException
        else:
            if self.network.layers[0].input is None:
                self.network.layers[0].input = self.x

        for layer in self.network.layers:
            if isinstance(layer, UnsupervisedLayer):
                layer.unsupervised_training(self.x, self.data.train_set_x)

    @timer(' Training')
    def train(self, unsupervised_training=True, save_mode=None, shuffle=True):
        """
        Training the network

        Parameters
        ----------
        unsupervised_training: `bool`, (default is True)
            pre-training of the unsupervised layers if any
        save_mode : {None, 'end', 'each'}
            None (default), model will not be saved unless name specified in the
            model definition.
            'end', model will only be saved at the end of the training
            'each', model will be saved each time the model is improved

        Returns
        -------
            report
        """
        start_time = timeit.default_timer()

        if self.data is None:
            raise NoDataFoundException
        else:
            if self.data.train_set_y.ndim == 1:
                self.y = T.ivector('y')  # the output labels are presented as 1D vector of[int] labels
            else:
                self.y = T.matrix('y')

        if self.network is None:
            raise NoNetworkFoundException
        else:
            if self.network.layers[0].input is None:
                self.network.layers[0].input = self.x

        if save_mode is not None:
            if save_mode not in ['end', 'each']:
                self.save_mode = 'end'
            else:
                self.save_mode = save_mode
            if self.file is None:
                import datetime
                self.file = self.name + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.ym'

        if self.file is not None and save_mode is None:
            self.save_mode = 'end'

        self.report['test_values'] = []
        self.report['validation_values'] = []

        if unsupervised_training and self.network.has_unsupervised_layer:
            self.pretrain()

        n_train_batches = self.data.train_set_x.get_value(borrow=True).shape[0] / self.hp.batch_size
        n_valid_batches = self.data.valid_set_x.get_value(borrow=True).shape[0] / self.hp.batch_size
        n_test_batches = self.data.test_set_x.get_value(borrow=True).shape[0] / self.hp.batch_size

        train_idx = np.arange(n_train_batches * self.hp.batch_size, dtype='int32')

        ################################################
        # cost
        cost = T.mean(self.objective(prediction=self.network.get_output(stochastic=True), target=self.y))
        # add regularisation
        cost += self.network.reguls

        ################################################
        # Updates
        # updates of the model as a list of (variable, update expression) pairs
        update_param = {}
        if hasattr(self.hp, 'learning_rate'):
            update_param['learning_rate'] = self.hp.learning_rate
        if hasattr(self.hp, 'momentum'):
            update_param['momentum'] = self.hp.momentum
        if hasattr(self.hp, 'epsilon'):
            update_param['epsilon'] = self.hp.epsilon
        if hasattr(self.hp, 'rho'):
            update_param['rho'] = self.hp.rho
        if hasattr(self.hp, 'beta1'):
            update_param['beta1'] = self.hp.beta1
        if hasattr(self.hp, 'beta2'):
            update_param['beta2'] = self.hp.beta1
        updates = self.updates(cost, self.network.params, **update_param)

        ################################################
        # Validation & Test functions
        error = categorical_error(self.network.get_output(stochastic=False), self.y)

        ################################################
        # Compiling functions for training, validating and testing the model
        logger.info('... Compiling the model')
        train_model = theano.function(inputs=[self.index, self.epoch_index], outputs=cost, updates=updates, name='train',
                                      givens={self.x: self.data.train_set_x[self.epoch_index[self.index * self.hp.batch_size: (self.index + 1) * self.hp.batch_size]],
                                              self.y: self.data.train_set_y[self.epoch_index[self.index * self.hp.batch_size: (self.index + 1) * self.hp.batch_size]]})

        validate_model = theano.function(inputs=[self.index], outputs=error, name='validate',
                                         givens={self.x: self.data.valid_set_x[self.index * self.hp.batch_size: (self.index + 1) * self.hp.batch_size],
                                                 self.y: self.data.valid_set_y[self.index * self.hp.batch_size: (self.index + 1) * self.hp.batch_size]})

        test_model = theano.function(inputs=[self.index], outputs=error, name='test',
                                     givens={self.x: self.data.test_set_x[self.index * self.hp.batch_size: (self.index + 1) * self.hp.batch_size],
                                             self.y: self.data.test_set_y[self.index * self.hp.batch_size: (self.index + 1) * self.hp.batch_size]})

        ################################################
        # Training
        logger.info('... Training the model')

        # early-stopping parameters
        patience = self.hp.patience     # look at this many batches regardless
        patience_increase = 2           # wait this much longer when a new best is found
        improvement_threshold = 0.995   # a relative improvement of this much is considered significant
        validation_frequency = min(n_train_batches, patience / 2)  # go through this many minibatches before checking the network

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        epoch = 0
        done_looping = False

        while (epoch < self.hp.n_epochs) and (not done_looping):
            epoch += 1
            if shuffle:
                np_rng.shuffle(train_idx)
            for minibatch_index in xrange(n_train_batches):
                # train
                minibatch_avg_cost = train_model(minibatch_index, train_idx)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)

                    logger.info('epoch %i, minibatch %i/%i, validation error %.3f %%' %
                                (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))
                    self.report['validation_values'].append((iter + 1, this_validation_loss * 100.))

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

                        logger.info('  epoch %i, minibatch %i/%i, test error of best model %.3f %%' %
                                    (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
                        self.report['test_values'].append((epoch, test_score * 100))

                        # save and overwrite each best model
                        if self.save_mode == 'each':
                            save_model(self)
                            logger.info(' Best model saved')

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()

        # save the final model
        if self.save_mode == 'end':
            save_model(self)
            logger.info(' Final model saved as : ' + self.file)

        logger.info('\n Optimization completed. ' + ('Early stopped at epoch: %i' % epoch)
                    if done_looping else 'Optimization completed. ' + ('Trained on all %i epochs' % epoch))

        logger.info(' Validation score of %.3f %% obtained at iteration %i, with test performance %.3f %%' %
                    (best_validation_loss * 100., best_iter + 1, test_score * 100.))

        # Report
        self.report['epoch'] = epoch
        self.report['early_stop'] = done_looping
        self.report['best_validation'] = best_validation_loss * 100.
        self.report['best_iter'] = best_iter + 1
        self.report['test_score'] = test_score * 100.
        self.report['training_duration'] = format_sec(end_time - start_time)

        return self.report

    def predict(self, X):
        if self.network.layers[0].input is None:
            self.network.layers[0].input = self.x
        prediction = self.network.get_output(stochastic=False)
        predict = theano.function(inputs=[self.x], outputs=prediction, name='predict')
        return predict(X)

    def to_conf(self, file=None):
        conf = {'model name': self.name,
                'hyperparameters': self.hp.to_conf(),
                'network': self.network.to_conf(),
                'updates': self.updates.__name__,
                'report': self.report,
                'file': self.file}
        if file is None:
            return conf
        else:
            with open(file, 'wb') as f:
                cPickle.dump(conf, f, cPickle.HIGHEST_PROTOCOL)

    def from_conf(self, conf=None, file = None):
        if file:
            with open(file, 'rb') as f:
                _conf = cPickle.load(f)
        else:
            _conf = conf.copy()
        self.name = _conf['model name']
        self.hp = yadll.hyperparameters.Hyperparameters()
        for k, v in _conf['hyperparameters'].iteritems():
            self.hp(k, v)
        self.network = yadll.network.Network()
        self.network.from_conf(_conf['network'])
        self.updates = getattr(yadll.updates, _conf['updates'])
        self.report = _conf['report']
        self.file = _conf['file']


