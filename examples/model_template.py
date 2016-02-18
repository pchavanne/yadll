#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os

import dl
from dl.hyperparameters import *
from dl.model import *

# load the data
datafile = 'mnist.pkl.gz'
if not os.path.isfile(datafile):
    import urllib
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, datafile)
data = dl.data.Data(datafile)

# create the model
model = dl.model.Model(name='mlp grid search', data=data)

# Hyperparameters
hp = Hyperparameters()
hp('batch_size', 500)
hp('n_epochs', 1000)
hp('learning_rate', 1)
#hp('momentum', 0.5)
hp('l1_reg', 0.00)
hp('l2_reg', 0.0001)
hp('patience', 10000)
# add the hyperparameters to the model
model.hp = hp

# Create connected layers
# Input layer
l_in = InputLayer(shape=(hp.batch_size, 28 * 28), input_var=model.x, name='Input')
# Dense Layer 1
l_hid1 = DenseLayer(incoming=l_in, nb_units=500, W=glorot_uniform, l1=hp.l1_reg,
                    l2=hp.l2_reg, activation=tanh, name='Hidden layer 1')
# Dense Layer 2
l_hid2 = DenseLayer(incoming=l_hid1, nb_units=500, W=glorot_uniform, l1=hp.l1_reg,
                    l2=hp.l2_reg, activation=tanh, name='Hidden layer 2')
# Logistic regression Layer
l_out = LogisticRegression(incoming=l_hid2, nb_class=10, l1=hp.l1_reg,
                           l2=hp.l2_reg, name='Logistic regression')

# Create network and add layers
net = Network('mlp')
net.add(l_in)
net.add(l_hid1)
net.add(l_hid2)
net.add(l_out)
# add the network to the model
model.network = net

# updates method
model.updates = dl.updates.rmsprop  # nesterov_momentum

# train the model
model.train()

print model.report
