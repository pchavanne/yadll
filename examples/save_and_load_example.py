#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
This example file show you how to creat a model, train it and save it.
You will save a model, save the parameters and save the configuration,
and rebuild the model.
"""
import os
import yadll
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# Load the data
datafile = 'mnist.pkl.gz'
if not os.path.isfile(datafile):
    import urllib
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    print 'Downloading data from %s' % origin
    urllib.urlretrieve(origin, datafile)
data = yadll.data.Data(datafile)

# Create the model. We provide a file name so the model will be saved at the end of the training
model = yadll.model.Model(name='mlp with dropout', data=data, file='best_model.ym')

# Hyperparameters
hp = yadll.hyperparameters.Hyperparameters()
hp('batch_size', 500)
hp('n_epochs', 1000)
hp('learning_rate', 0.9)
hp('l1_reg', 0.001)
hp('l2_reg', 0.00001)
hp('patience', 1000)

# Add the hyperparameters to the model
model.hp = hp

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
l_hid2 = yadll.layers.DenseLayer(incoming=l_dro2, n_units=250, W=yadll.init.glorot_uniform,
                                 l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu)
# Logistic regression Layer
l_out = yadll.layers.LogisticRegression(incoming=l_hid2, n_class=10)

# Create network and add layers
net = yadll.network.Network('2 layers mlp with dropout')
net.add(l_in)
net.add(l_dro1)
net.add(l_hid1)
net.add(l_dro2)
net.add(l_hid2)
net.add(l_out)

# Add the network to the model
model.network = net

# Updates method
model.updates = yadll.updates.sgd

# Saving configuration of the model. Model doesn't have to be trained
conf = model.to_conf()    # get the configuration
model.to_conf('conf.yc')  # or save it to file .yc by convention

# Train the model and save it to file at each best
model.train(save_mode='each')

# Saving network parameters after training
net.save_params('net_params.yp')

# Make prediction
# We can test it on some examples from test
test_set_x = data.test_set_x.get_value()
test_set_y = data.test_set_y.eval()

predicted_values = model.predict(test_set_x[:30])

print ("Model predicted values for the first 30 examples in test set:")
print predicted_values
print test_set_y[:30]

##########################################################################
# Loading model from file
model_2 = yadll.model.load_model('best_model.ym')
# model is ready to use we can make prediction directly.
# Watch out this not the proper way of saving models.
predicted_values_2 = model_2.predict(test_set_x[:30])
print ("Model 2 predicted values for the first 30 examples in test set:")
print predicted_values_2
print test_set_y[:30]

##########################################################################
# Recreate model and load parameters
model_3 = yadll.model.Model()
l_in = yadll.layers.InputLayer(input_shape=(hp.batch_size, 28 * 28))
l_dro1 = yadll.layers.Dropout(incoming=l_in, corruption_level=0.5)

l_hid1 = yadll.layers.DenseLayer(incoming=l_dro1, n_units=500, W=yadll.init.glorot_uniform,
                                 l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu)
l_dro2 = yadll.layers.Dropout(incoming=l_hid1, corruption_level=0.25)
l_hid2 = yadll.layers.DenseLayer(incoming=l_dro2, n_units=250, W=yadll.init.glorot_uniform,
                                 l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu)
l_out = yadll.layers.LogisticRegression(incoming=l_hid2, n_class=10)

# Create network and add layers
net = yadll.network.Network('2 layers mlp with dropout')
net.add(l_in)
net.add(l_dro1)
net.add(l_hid1)
net.add(l_dro2)
net.add(l_hid2)
net.add(l_out)
model_3.network = net
# Network as been re-created so parameters has just been initialized
# Let's try prediction with this network.
predicted_values_3 = model_3.predict(test_set_x[:30])
print ("Model 3 without loading parameters values for the first 30 examples in test set:")
print predicted_values_3
print test_set_y[:30]
# Now let's load parameters
model_3.network.load_params('net_params.yp')
# And try predicting again
predicted_values_3 = model_3.predict(test_set_x[:30])
print ("Model 3 after loading parameters values for the first 30 examples in test set:")
print predicted_values_3
print test_set_y[:30]

##########################################################################
# Reconstruction the model from configuration and load parameters
model_4 = yadll.model.Model()
model_4.from_conf(conf)         # load from conf obj
model_5 = yadll.model.Model()
model_5.from_conf(file='conf.yc')    # load from conf file

model_4.network.load_params('net_params.yp')
model_5.network.load_params('net_params.yp')

predicted_values_4 = model_4.predict(test_set_x[:30])
print ("Model 4 after loading parameters values for the first 30 examples in test set:")
print predicted_values_4
print test_set_y[:30]

predicted_values_5 = model_5.predict(test_set_x[:30])
print ("Model 5 after loading parameters values for the first 30 examples in test set:")
print predicted_values_5
print test_set_y[:30]
