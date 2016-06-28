.. _tutorial:

========
Tutorial
========

Build your first network
------------------------

Let's build our first MLP with dropout on the MNIST example.
We will first import yadll and configure a basic logger.

.. code-block:: python

    import os

    from yadll.model import Model
    from yadll.data import Data
    from yadll.hyperparameters import *
    from yadll.updates import *
    from yadll.network import Network
    from yadll.layers import *

    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

Then we will load the MNIST dataset (or download it) and create a `Data` instance

.. code-block:: python

    # load the data
    datafile = 'mnist.pkl.gz'
    if not os.path.isfile(datafile):
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, datafile)
    data = Data(datafile)

We create a Model, that is the class that hold the data, the network,
the hyperparameters and the updates function.

.. code-block:: python

    # create the model
    model = Model(name='MLP 2 layers with dropout', data=data)

We define the hyperparameters of the model and add it to our model object.

.. code-block:: python

    # Hyperparameters
    hp = Hyperparameters()
    hp('batch_size', 500)
    hp('n_epochs', 1000)
    hp('learning_rate', 0.1)
    hp('l1_reg', 0.00)
    hp('l2_reg', 0.0000)
    hp('patience', 10000)

    # add the hyperparameters to the model
    model.hp = hp

We now create each layers of the network and add them to the network object.

.. code-block:: python

    # Create connected layers
    # Input layer
    l_in = InputLayer(shape=(hp.batch_size, 28 * 28), input_var=model.x, name='Input')
    # Dropout Layer
    l_dro1 = Dropout(incoming=l_in, corruption_level=0.4, name='Dropout 1')
    # Dense Layer
    l_hid1 = DenseLayer(incoming=l_dro1, nb_units=500, W=glorot_uniform,
                        activation=relu, name='Hidden layer 1')
    # Dropout Layer
    l_dro2 = Dropout(incoming=l_hid1, corruption_level=0.2, name='Dropout 2')
    # Dense Layer
    l_hid2 = DenseLayer(incoming=l_dro2, nb_units=500, W=glorot_uniform,
                        activation=relu, name='Hidden layer 2')
    # Logistic regression Layer
    l_out = LogisticRegression(incoming=l_hid2, nb_class=10, name='Logistic regression')

    # Create network and add layers
    net = Network('dropout')
    net.add(l_in)
    net.add(l_dro1)
    net.add(l_hid1)
    net.add(l_dro2)
    net.add(l_hid2)
    net.add(l_out)

We add the network and the updates function to the model and train the model.

.. code-block:: python

    # add the network to the model
    model.network = net

    # add the updates method
    model.updates = sgd  # Stochastic Gradient Descent

    # train the model
    model.train()

Here is the output when trained on NVIDIA Geforce Titan X card:

.. code-block:: text

    epoch 998, minibatch 100/100, validation error 1.420 %
    epoch 999, minibatch 100/100, validation error 1.370 %
    epoch 1000, minibatch 100/100, validation error 1.350 %
    Optimization completed. Trained on all 1000 epochs
     Validation score of 1.290 % obtained at iteration 68700, with test performance 1.290 %
     Training MLP 2 layers with dropout took 03 m 12 s


Run the examples
----------------
Different networks are tested on MNIST dataset on the examples/mnist_dl.py

* Logisitic Regression
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
* Recurrent Neural Networks
* Long Short-Term Memory

You can get the list of all available networks:

.. code-block:: bash

  python mnist_dl.py --network_list


Trainning a model for example lenet5:

.. code-block:: bash

  python mnist_dl.py lenet5


grid search on the hyperparameters:

.. code-block:: bash

  python hp_grid_search.py