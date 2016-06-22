.. _installation:

============
Installation
============

Prerequisites
=============

We assume your are running a linux system.
We assume you have Python >=2.7 with numpy and pandas.
You can install it from Anaconda_ from Continuum Analytics

.. _Anaconda: https://www.continuum.io/downloads

We assume you have pip:

.. code-block:: bash

  sudo apt-get install pip

We assume you have `Installed Theano`_.

.. _`Installed Theano`: http://deeplearning.net/software/theano/install.html


Installation
------------
The easiest way to install yadll is
with the Python package manager ``pip``:

.. code-block:: bash

  git clone git@github.com:pchavanne/yadll.git
  cd yadll
  pip install -e .


GPU Support
===========
If you have a NVIDA card you can set up CUDA and have Theano to use your GPU.
See the 'Using the GPU' in the `installing Theano`_ instruction.

.. _`installing Theano`: http://deeplearning.net/software/theano/install.html