.. yadll documentation master file, created by
   sphinx-quickstart on Thu Jun 16 15:24:44 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to Yadll
================

**Y**\ et **a**\ nother **d**\ eep **l**\ earning **l**\ ab.

This is an ultra light deep learning framework written in Python and based on Theano_.
It allows you to very quickly start building Deep Learning models. It was originally the code, notes and references I gathered when following the
`Theano's Deep Learning Tutorials`_ tutorial then I used Lasagne_, keras_ or blocks_ and restructured this code based on it.

If you are looking for mature deep learning APIs I would recommend Lasagne_, keras_ or blocks_ in stead of yadll, they are well documented and contributed projects.

Read the documentation at `Read the doc`_

.. _Theano: https://github.com/Theano/Theano
.. _`Theano's Deep Learning Tutorials`: http://deeplearning.net/tutorial/contents.html
.. _Lasagne: https://github.com/Lasagne/Lasagne
.. _keras: https://github.com/fchollet/keras
.. _blocks: https://github.com/mila-udem/blocks
.. _`Read the doc`: http://yadll.readthedocs.io/en/latest/


User Guide
----------

The Yadll user guide explains how to install Yadll, how to build and train
neural networks using many different models on mnist.

.. toctree::
  :maxdepth: 2

  user/installation
  user/tutorial
  user/layers


API Reference
-------------

Referencees on functions, classes or methodes, with notes and references.

.. toctree::
  :maxdepth: 2

  modules/model
  modules/network
  modules/data
  modules/hyperparameters
  modules/layers
  modules/updates
  modules/init
  modules/activation
  modules/objectives
  modules/utils


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/pchavanne/yadll