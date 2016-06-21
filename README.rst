.. image:: https://travis-ci.org/pchavanne/yadll.svg
    :target: https://travis-ci.org/pchavanne/yadll

.. image:: https://coveralls.io/repos/github/pchavanne/yadll/badge.svg?branch=master
    :target: https://coveralls.io/github/pchavanne/yadll?branch=master

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/pchavanne/yadll/blob/master/LICENSE

.. image:: https://readthedocs.org/projects/yadll/badge/
    :target: http://yadll.readthedocs.io/en/latest/


=====
Yadll
=====


**Y**\ et **a**\ nother **d**\ eep **l**\ earning **l**\ ab.

This is an ultra light deep learning framework written in Python and based on Theano_.
It allows you to very quickly start building Deep Learning models. It was originally the code, notes and references I gathered when following the
`Theano's Deep Learning Tutorials`_ tutorial then I used Lasagne_ and keras_ and restructured this code based on it.

If you are looking for a light deep learning API I would recommend using Lasagne_ or keras_ in stead of yadll, both are mature, well documented and contributed projects.

Read the documentation at `Read the doc`_

.. _Theano: https://github.com/Theano/Theano
.. _`Theano's Deep Learning Tutorials`: http://deeplearning.net/tutorial/contents.html
.. _Lasagne: https://github.com/Lasagne/Lasagne
.. _keras: https://github.com/fchollet/keras
.. _`Read the doc`: http://yadll.readthedocs.io/en/latest/


Its main features are:

* **Layers**:

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
  * RNN
  * LSTM

* **Optimisation**:

  * Sgd
  * Momentum
  * Nesterov momentum
  * Adagrad
  * Adadelta
  * Rmsprop
  * Hessian Free


* **Hyperparameters grid search**

Installation
------------
**yadll** uses the following dependencies:

* **Python 2.7**
* scipy
* theano

.. code-block:: bash

  git clone git@github.com:pchavanne/yadll.git
  cd yadll
  pip install -e .

Example
-------

Different networks tested on mnist:

* logisitic Regression
* Multi Layer Perceptron
* MLP with dropout
* MLP with dropconnect
* Conv Pool
* LeNet-5
* Autoencoder
* Denoising Autoencoder
* Gaussian Denoising Autoencoder
* Contractive Denoising Autoencoder
* Stacked Denoising Autoencoder
* Restricted Boltzmann Machine
* Deep Belief Network
* Recurent Neural Networks
* Long Short-Term Memory

get the list of available networks:

.. code-block:: bash

  python mnist_dl.py --network_list


trainning a model for example lenet5:

.. code-block:: bash

  python mnist_dl.py lenet5


grid search on the hyperparameters:

.. code-block:: bash

  python hp_grid_search.py