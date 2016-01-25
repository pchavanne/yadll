.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/pchavanne/dl/blob/master/LICENSE

dl
==

dl is my personal ultra light deep learning framework based on **Theano**.
It is highly inspired by **Lasagne** and **keras** I would therefore recommend
considering one or the other in stead of dl.


Its main features are:

* Layers: Base, Input, Dense, Dropout, Dropconnect
* optimisation: sgd, momentum, Nesterov momentum, adagrad, adadelta, rmsprop


Installation
------------
dl uses the following dependencies:

pip install git+git@github.com:pchavanne/dl.git


Documentation
-------------

not yet implemented


Example
-------

different network tested on mnist:
    - Linear Regression
    - MLP
    - MLP with dropout
    - MLP with dropconnect
