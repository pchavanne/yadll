#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This example show you how to train an LSTM for text generation.
"""
import numpy as np
import yadll

import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

alphabet = 'abcdefghijklmnopqrstuvwxyz'
sequence_length = 2
number_of_chars = len(alphabet)
# load the data
data = yadll.data.Data(yadll.data.alphabet_loader(sequence_length))

# create the model
model = yadll.model.Model(name='lstm', data=data)

# Hyperparameters
hp = yadll.hyperparameters.Hyperparameters()
hp('batch_size', 1)
hp('n_epochs', 100)
hp('patience', 1000)

# add the hyperparameters to the model
model.hp = hp

# Create connected layers
# Input layer
l_in = yadll.layers.InputLayer(input_shape=(hp.batch_size, sequence_length, number_of_chars))
# LSTM 1
l_lstm1 = yadll.layers.BNLSTM(incoming=l_in, n_units=16, last_only=False)
# LSTM 2
l_lstm2 = yadll.layers.BNLSTM(incoming=l_lstm1, n_units=16)
# Logistic regression Layer

l_out = yadll.layers.LogisticRegression(incoming=l_lstm2, n_class=number_of_chars)

# Create network and add layers
net = yadll.network.Network('stacked lstm')
net.add(l_in)
net.add(l_lstm1)
net.add(l_lstm2)
net.add(l_out)

# add the network to the model
model.network = net
# updates method
model.updates = yadll.updates.rmsprop

# train the model and save it to file at each best
model.compile(compile_arg='all')
model.train()

# prime the model with 'ab' sequence and let it generate the learned alphabet
sentence = alphabet[:sequence_length]
generated = sentence
for iteration in range(number_of_chars - sequence_length):
    x = np.zeros((1, sequence_length, number_of_chars))
    for t, char in enumerate(sentence):
        x[0, t, ord(char) - ord('a')] = 1.
    preds = model.predict(np.asarray(x, dtype='float32'))[0]
    next_char = chr(np.argmax(preds) + ord('a'))
    generated += next_char
    sentence = sentence[1:] + next_char

# check that it did generate the alphabet correctly
assert(generated == alphabet)
x = model.data.test_set_x.get_value()[0:3]
model.predict(x)
chr(np.argmax(model.predict(x)) + ord('a'))
