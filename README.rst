.. image:: https://travis-ci.org/pchavanne/dl.svg
    :target: https://travis-ci.org/pchavanne/dl

.. image:: https://img.shields.io/coveralls/pchavanne/dl.svg
    :target: https://coveralls.io/r/pchavanne/dl

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/pchavanne/dl/blob/master/LICENSE


dl
==

**dl** is an ultra light deep learning framework based on Theano_.
If you are looking for a light deep learning API I would recommend Lasagne_ or keras_ in stead of dl.

.. _Theano: https://github.com/Theano/Theano
.. _Lasagne: https://github.com/Lasagne/Lasagne
.. _keras: https://github.com/fchollet/keras

Its main features are:

* Layer:

  * Input Layer
  * Dropout Layer
  * Pool Layer
  * Conv Layer:

    * ConvPool Layer
  * Dense Layer:

    * Logistic Regression
    * Dropconnect
    * Unsupervised Layer:

      * Autoencoder (denoising autoencoder)
      * Restricted Boltzmann Machine
  * LSTM


* Optimisation:

  * Sgd
  * Momentum
  * Nesterov momentum
  * Adagrad
  * Adadelta
  * Rmsprop
  * Hessian Free


* Hyperparameters grid search

Installation
------------
**dl** uses the following dependencies:

* **Python 2.7**
* scipy
* theano

.. code-block:: bash

  git clone git+git@github.com:pchavanne/dl.git
  cd dl
  pip install -e .


Example
-------

Different networks tested on mnist:

* Linear Regression
* MLP
* MLP with dropout
* MLP with dropconnect
* Conv Pool
* LeNet-5
* Autoencoder
* Denoising Autoencoder
* Stacked Denoising Autoencoder
* Restricted Boltzmann Machine
* Deep Belief Network
* Convolutional Network
* LSTM

get the list of available networks:

.. code-block:: bash

  python mnist-dl.py --network_list


train the model mlp

.. code-block:: bash

  python mnist-dl.py mlp