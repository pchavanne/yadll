#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
This example will show you the difference between the updates function:
    - sgd: Stochastic Gradient Descent
    - momentum: Stochastic Gradient Descent with momentum
    - nesterov_momentum: Stochastic Gradient Descent with Nesterov momentum
    - adagrad: Adaptive gradient descent
    - rmsprop: scaling with the Root mean square of the gradient
    - adadelta: adaptive learning rate
    - adam: Adaptive moment gradient descent
    - adamax: adam with infinity norm
"""
import os
import yadll
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# load the data
datafile = 'mnist.pkl.gz'
if not os.path.isfile(datafile):
    import urllib
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, datafile)
data = yadll.data.Data(datafile)

updates = OrderedDict([
    ('sgd', [['learning_rate', 0.1, [0.001, 0.01, 0.1]]]),
    ('momentum', [['learning_rate', 0.1, [0.001, 0.01, 0.1]],
                  ['momentum', 0.9, [0.85, 0.9, 0.95, 0.99]]]),
    ('nesterov_momentum', [['learning_rate', 0.1, [0.001, 0.01, 0.1]],
                           ['momentum', 0.9, [0.85, 0.9, 0.95, 0.99]]]),
    ('adagrad', [['learning_rate', 0.1, [0.001, 0.01, 0.1]]]),
    ('rmsprop', [['learning_rate', 0.1, [0.001, 0.01, 0.1]]]),
    ('adadelta', [['learning_rate', 0.1, [0.001, 0.01, 0.1]]]),
    ('adam', [['learning_rate', 0.1, [0.001, 0.01, 0.1]]]),
    ('adamax', [['learning_rate', 0.1, [0.001, 0.01, 0.1]]]),
])


def get_hps():
    # Hyperparameters
    hps = yadll.hyperparameters.Hyperparameters()
    hps('batch_size', 50)
    hps('n_epochs', 500)
    hps('l1_reg', 0.001)
    hps('l2_reg', 0.00001)
    hps('patience', 5000)
    return hps


def get_model(hp):
    # create the model
    model = yadll.model.Model(name='mlp with dropout', data=data)
    # Create connected layers
    # Input layer
    l_in = yadll.layers.InputLayer(input_shape=(hp.batch_size, 28 * 28))
    # Dropout Layer 1
    l_dro1 = yadll.layers.Dropout(incoming=l_in, corruption_level=0.5)
    # Dense Layer 1
    l_hid1 = yadll.layers.DenseLayer(incoming=l_dro1, n_units=500, W=yadll.init.glorot_uniform,
                                     l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu)
    # Dropout Layer 2
    l_dro2 = yadll.layers.Dropout(incoming=l_hid1, corruption_level=0.25)
    # Dense Layer 2
    l_hid2 = yadll.layers.DenseLayer(incoming=l_dro2, n_units=500, W=yadll.init.glorot_uniform,
                                     l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu)
    # Logistic regression Layer
    l_out = yadll.layers.LogisticRegression(incoming=l_hid2, n_class=10, l1=hp.l1_reg, l2=hp.l2_reg)

    # Create network and add layers
    net = yadll.network.Network('2 layers mlp with dropout')
    net.add(l_in)
    net.add(l_dro1)
    net.add(l_hid1)
    net.add(l_dro2)
    net.add(l_hid2)
    net.add(l_out)

    # add the network to the model
    model.network = net

    # add the hyperparameters to the model
    model.hp = hp

    return model

report = list()

for update, hyperparams in updates.iteritems():
    hps = get_hps()
    for hyperparam in hyperparams:
        hps(*hyperparam)
    for hp in hps:
        model = get_model(hp)
        model.updates = getattr(yadll.updates, update)
        model.train()
        r = list()
        r.append(update)
        for hyperparam in hyperparams:
            r.append(hyperparam[0])
            r.append(hp.hp_value[hyperparam[0]])
        r.append('epoch')
        r.append(model.report['epoch'])
        r.append('early_stop')
        r.append(model.report['early_stop'])
        r.append('best_validation')
        r.append(round(model.report['best_validation'], 2))
        r.append('best_iter')
        r.append(model.report['best_iter'])
        r.append('test_score')
        r.append(round(model.report['test_score'], 2))
        r.append('training_duration')
        r.append(model.report['training_duration'])
        report.append(r)
        print report
        with open('report', 'w') as f:
            f.writelines(' '.join(str(e) for e in r))

