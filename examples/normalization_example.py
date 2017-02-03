#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
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
hp('batch_size', 500)
hp('n_epochs', 1000)
hp('learning_rate', 0.01)
hp('momentum', 0.5)
hp('l1_reg', 0.00)
hp('l2_reg', 0.0000)
hp('patience', 10000)

# add the hyperparameters to the model
model.hp = hp

# Create connected layers
# Input layer
l_in = yadll.layers.InputLayer(input_shape=(hp.batch_size, 28 * 28), name='Input')
# Dense Layer 1
l_hid1 = yadll.layers.DenseLayer(incoming=l_in, n_units=500, W=yadll.init.glorot_uniform,
                                 l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu,
                                 name='Hidden layer 1')
# Dense Layer 2
l_hid2 = yadll.layers.DenseLayer(incoming=l_hid1, n_units=500, W=yadll.init.glorot_uniform,
                                 l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu,
                                 name='Hidden layer 2')
# Logistic regression Layer
l_out = yadll.layers.LogisticRegression(incoming=l_hid2, n_class=10, l1=hp.l1_reg,
                                        l2=hp.l2_reg, name='Logistic regression')

# Create network and add layers
net = yadll.network.Network('2 layers mlp with dropout')
net.add(l_in)
net.add(l_hid1)
net.add(l_hid2)
net.add(l_out)

# add the network to the model
model.network = net

# updates method
model.updates = yadll.updates.adagrad

# train the model and save it to file at each best
model_report = model.train()


######################################################################################
# Basic mlp with batch normalization model
# create the model
model_BN = yadll.model.Model(name='mlp with batch normalization', data=data)

# # Hyperparameters
# hp = yadll.hyperparameters.Hyperparameters()
# hp('batch_size', 500)
# hp('n_epochs', 1000)
# hp('learning_rate', 0.9)
# hp('momentum', 0.5)
# hp('l1_reg', 0.00)
# hp('l2_reg', 0.0000)
# hp('patience', 10000)

# add the hyperparameters to the model
model_BN.hp = hp

# Create connected layers
# Input layer
l_in = yadll.layers.InputLayer(input_shape=(hp.batch_size, 28 * 28), name='Input')
# Batch Normalization 1
l_bn1 = yadll.layers.BatchNormalization(incoming=l_in, name='Batch Normalization 1')
# Dense Layer 1
l_hid1 = yadll.layers.DenseLayer(incoming=l_bn1, n_units=500, W=yadll.init.glorot_uniform,
                                 l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu,
                                 name='Hidden layer 1')
# Batch Normalization 2
l_bn2 = yadll.layers.BatchNormalization(incoming=l_hid1, name='Batch Normalization 2')
# Dense Layer 2
l_hid2 = yadll.layers.DenseLayer(incoming=l_bn2, n_units=500, W=yadll.init.glorot_uniform,
                                 l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu,
                                 name='Hidden layer 2')
# Batch Normalization 3
l_bn3 = yadll.layers.BatchNormalization(incoming=l_hid2, name='Batch Normalization 3')
# Logistic regression Layer
l_out = yadll.layers.LogisticRegression(incoming=l_bn3, n_class=10, l1=hp.l1_reg,
                                        l2=hp.l2_reg, name='Logistic regression')

# Create network and add layers
net = yadll.network.Network('mlp with batch normalization')
net.add(l_in)
net.add(l_bn1)
net.add(l_hid1)
net.add(l_bn2)
net.add(l_hid2)
net.add(l_bn3)
net.add(l_out)

# add the network to the model
model_BN.network = net

# updates method
model_BN.updates = yadll.updates.adagrad

# train the model and save it to file at each best
model_BN_report = model.train()
