#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import yadll
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# load the data
data = yadll.data.Data(yadll.data.mnist_loader())

######################################################################################
# Basic mlp model
# create the model
model = yadll.model.Model(name='mlp', data=data)

# Hyperparameters
hp = yadll.hyperparameters.Hyperparameters()
hp('batch_size', 512)
hp('n_epochs', 1000)
hp('patience', 10000)

# add the hyperparameters to the model
model.hp = hp

# Create connected layers
# Input layer
l_in = yadll.layers.InputLayer(input_shape=(hp.batch_size, 28 * 28))
# Dense Layer 1
l_hid1 = yadll.layers.DenseLayer(incoming=l_in, n_units=500)
# Dense Layer 2
l_hid2 = yadll.layers.DenseLayer(incoming=l_hid1, n_units=500)
# Logistic regression Layer
l_out = yadll.layers.LogisticRegression(incoming=l_hid2, n_class=10)

# Create network and add layers
net = yadll.network.Network('2 layers mlp')
net.add(l_in)
net.add(l_hid1)
net.add(l_hid2)
net.add(l_out)

# add the network to the model
model.network = net

# updates method
model.updates = yadll.updates.rmsprop

# train the model and save it to file at each best
model_report = model.train()


######################################################################################
# Basic mlp with batch normalization model
# create the model
model_bn = yadll.model.Model(name='mlp with batch normalization', data=data, hyperparameters=hp)

# Create connected layers
# Input layer
l_in = yadll.layers.InputLayer(input_shape=(hp.batch_size, 28 * 28))
# Batch Normalization 1
l_bn1 = yadll.layers.BatchNormalization(incoming=l_in)
# Dense Layer 1
l_hid1 = yadll.layers.DenseLayer(incoming=l_bn1, n_units=500, activation=yadll.activations.linear)
# Batch Normalization 2
l_bn2 = yadll.layers.BatchNormalization(incoming=l_hid1)
# Activation 1
l_act1 = yadll.layers.Activation(incoming=l_bn2, activation=yadll.activations.relu)
# Dense Layer 2
l_hid2 = yadll.layers.DenseLayer(incoming=l_act1, n_units=500, activation=yadll.activations.linear)
# Batch Normalization 3
l_bn3 = yadll.layers.BatchNormalization(incoming=l_hid2)
# Activation 2
l_act2 = yadll.layers.Activation(incoming=l_bn3, activation=yadll.activations.relu)
# Logistic regression Layer
l_out = yadll.layers.LogisticRegression(incoming=l_act2, n_class=10)

# Create network and add layers
net = yadll.network.Network('mlp with batch normalization')
net.add(l_in)
net.add(l_bn1)
net.add(l_hid1)
net.add(l_bn2)
net.add(l_act1)
net.add(l_hid2)
net.add(l_bn3)
net.add(l_act2)
net.add(l_out)

# add the network to the model
model_bn.network = net

# updates method
model_bn.updates = yadll.updates.rmsprop

# train the model and save it to file at each best
model_bn_report = model_bn.train()
