#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import yadll
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# load the data
datafile = 'mnist.pkl.gz'
if not os.path.isfile(datafile):
    import urllib
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, datafile)
data = yadll.data.Data(datafile)

# create the model
model = yadll.model.Model(name='mlp with dropout', data=data, file='best_model.ym')

# Hyperparameters
hp = yadll.hyperparameters.Hyperparameters()
hp('batch_size', 500)
hp('n_epochs', 1000)
hp('learning_rate', 0.9)
hp('momentum', 0.5)
hp('l1_reg', 0.00)
hp('l2_reg', 0.0000)
hp('patience', 10000)

# add the hyperparameters to the model
model.hp = hp

# Create connected layers
# Input layer
l_in = yadll.layers.InputLayer(input_shape=(hp.batch_size, 28 * 28), name='Input')
# Dropout Layer 1
l_dro1 = yadll.layers.Dropout(incoming=l_in, corruption_level=0.4, name='Dropout 1')
# Dense Layer 1
l_hid1 = yadll.layers.DenseLayer(incoming=l_dro1, nb_units=500, W=yadll.init.glorot_uniform,
                                 l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activation.relu,
                                 name='Hidden layer 1')
# Dropout Layer 2
l_dro2 = yadll.layers.Dropout(incoming=l_hid1, corruption_level=0.2, name='Dropout 2')
# Dense Layer 2
l_hid2 = yadll.layers.DenseLayer(incoming=l_dro2, nb_units=500, W=yadll.init.glorot_uniform,
                                 l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activation.relu,
                                 name='Hidden layer 2')
# Logistic regression Layer
l_out = yadll.layers.LogisticRegression(incoming=l_hid2, nb_class=10, l1=hp.l1_reg,
                                        l2=hp.l2_reg, name='Logistic regression')

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

# updates method
model.updates = yadll.updates.adagrad

# train the model and save it to file at each best
model.train()

# saving network paramters
net.save_params('net_params.yp')

# make prediction
# We can test it on some examples from test
test_set_x = data.test_set_x.get_value()
test_set_y = data.test_set_y.eval()

predicted_values = model.predict(test_set_x[:30])

print ("Model 1, predicted values for the first 30 examples in test set:")
print predicted_values
print test_set_y[:30]

# loading saved model
print ("Loading model from file")
# load the saved model
model2 = yadll.model.load_model('best_model.ym')

predicted_values2 = model2.predict(test_set_x[:30])
print ("Model 2, predicted values for the first 30 examples in test set:")
print predicted_values2
print test_set_y[:30]

# load network parameters
# first we recreate the network
# create the model
model3 = yadll.model.Model(name='mlp with dropout', data=data,)

# Hyperparameters
hp = yadll.hyperparameters.Hyperparameters()
hp('batch_size', 500)
hp('n_epochs', 1000)
hp('learning_rate', 0.1)
hp('momentum', 0.5)
hp('l1_reg', 0.00)
hp('l2_reg', 0.0000)
hp('patience', 10000)

# add the hyperparameters to the model
model3.hp = hp

# Create connected layers
# Input layer
l_in = yadll.layers.InputLayer(input_shape=(hp.batch_size, 28 * 28), name='Input')
# Dropout Layer 1
l_dro1 = yadll.layers.Dropout(incoming=l_in, corruption_level=0.4, name='Dropout 1')
# Dense Layer 1
l_hid1 = yadll.layers.DenseLayer(incoming=l_dro1, nb_units=500, W=yadll.init.glorot_uniform,
                                 l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activation.relu,
                                 name='Hidden layer 1')
# Dropout Layer 2
l_dro2 = yadll.layers.Dropout(incoming=l_hid1, corruption_level=0.2, name='Dropout 2')
# Dense Layer 2
l_hid2 = yadll.layers.DenseLayer(incoming=l_dro2, nb_units=500, W=yadll.init.glorot_uniform,
                                 l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activation.relu,
                                 name='Hidden layer 2')
# Logistic regression Layer
l_out = yadll.layers.LogisticRegression(incoming=l_hid2, nb_class=10, l1=hp.l1_reg,
                                        l2=hp.l2_reg, name='Logistic regression')

# Create network and add layers
net3 = yadll.network.Network('2 layers mlp with dropout')
net3.add(l_in)
net3.add(l_dro1)
net3.add(l_hid1)
net3.add(l_dro2)
net3.add(l_hid2)
net3.add(l_out)

# load params
net3.load_params('net_params.yp')

# add the network to the model
model3.network = net3

predicted_values3 = model3.predict(test_set_x[:30])
print ("Model 3, predicted values for the first 30 examples in test set:")
print predicted_values3
print test_set_y[:30]