:mod:`yadll.layers`

Layers
======

The Layers classes implement one layer of neural network of different types.
the ::class:Layer is the mother class of all the layers and has to be inherited
by any new layer.

.. automodule:: yadll.layers

.. autosummary::

   Layer
   InputLayer
   ReshapeLayer
   FlattenLayer
   DenseLayer
   UnsupervisedLayer
   LogisticRegression
   Dropout
   Dropconnect
   PoolLayer
   ConvLayer
   ConvPoolLayer
   AutoEncoder
   RBM
   BatchNormalization
   RNN
   LSTM

.. inheritance-diagram:: yadll.layers

Detailed description
--------------------

.. autoclass:: Layer
    :members:
.. autoclass:: InputLayer
    :members:
.. autoclass:: ReshapeLayer
    :members:
.. autoclass:: FlattenLayer
    :members:
.. autoclass:: DenseLayer
    :members:
.. autoclass:: UnsupervisedLayer
    :members:
.. autoclass:: LogisticRegression
    :members:
.. autoclass:: Dropout
    :members:
.. autoclass:: Dropconnect
    :members:
.. autoclass:: PoolLayer
    :members:
.. autoclass:: ConvLayer
    :members:
.. autoclass:: ConvPoolLayer
    :members:
.. autoclass:: AutoEncoder
    :members:
.. autoclass:: RBM
    :members:
.. autoclass:: BatchNormalization
    :members:
.. autoclass:: RNN
    :members:
.. autoclass:: LSTM
    :members:
