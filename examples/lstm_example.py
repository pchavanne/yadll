#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
This example show you how to train an LSTM for text generation.
"""
import os
import numpy as np
import yadll

import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# Creat the data
alphabet = 'abcdefghijklmnopqrstuvwxyz'
number_of_chars = len(alphabet)
sequence_length = 2
sentences = [alphabet[i: i + sequence_length] for i in range(len(alphabet) - sequence_length)]
next_chars = [alphabet[i + sequence_length] for i in range(len(alphabet) - sequence_length)]

# Transform sequences and labels into 'one-hot' encoding
X = np.zeros((len(sentences), sequence_length, number_of_chars), dtype=np.bool)
y = np.zeros((len(sentences), number_of_chars), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, ord(char) - ord('a')] = 1
    y[i, ord(next_chars[i]) - ord('a')] = 1
data = yadll.data.Data(data=[(X, y), (X, y), (X, y)])

# create the model
model = yadll.model.Model(name='lstm', data=data)

# Hyperparameters
hp = yadll.hyperparameters.Hyperparameters()
hp('batch_size', 1)
hp('n_epochs', 60)

# add the hyperparameters to the model
model.hp = hp

# Create connected layers
# Input layer
l_in = yadll.layers.InputLayer(input_shape=(hp.batch_size, sequence_length, number_of_chars))
# LSTM 1
l_lstm1 = yadll.layers.LSTM(incoming=l_in, n_units=16, last_only=False)
# LSTM 2
l_lstm2 = yadll.layers.LSTM(incoming=l_lstm1, n_units=16)
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
model.updates = yadll.updates.adam

# train the model and save it to file at each best
model.train()

# prime the model with 'ab' sequence and let it generate the learned alphabet
sentence = alphabet[:sequence_length]
generated = sentence
for iteration in range(number_of_chars - sequence_length):
    x = np.zeros((1, sequence_length, number_of_chars))
    for t, char in enumerate(sentence):
        x[0, t, ord(char) - ord('a')] = 1.
    preds = model.predict(x)[0]
    next_char = chr(np.argmax(preds) + ord('a'))
    generated += next_char
    sentence = sentence[1:] + next_char

# check that it did generate the alphabet correctly
assert(generated == alphabet)
