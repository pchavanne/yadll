.. _tutorial:

========
Tutorial
========

Building and training your first network
----------------------------------------

Let's build our first MLP with dropout on the MNIST example.
to run this example, just do:

.. code-block:: bash

    cd /yadll.examples
    python model_template.py

We will first import yadll and configure a basic logger.

.. code-block:: python

    import os

    import yadll

    import logging

    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

Then we load the MNIST dataset (or download it) and create a
:class:`yadll.data.Data` instance that will hold the data

.. code-block:: python

    # load the data
    datafile = 'mnist.pkl.gz'
    if not os.path.isfile(datafile):
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, datafile)
    data = yadll.data.Data(datafile)

We now create a :class:`yadll.model.Model`, that is the class that contain
the data, the network, the hyperparameters and the updates function. As a file
name is provided, the model will be saved (see Saving/loading models).

.. code-block:: python

    # create the model
    model = yadll.model.Model(name='mlp with dropout', data=data, file='best_model.ym')

We define the hyperparameters of the model and add it to our model object.

.. code-block:: python

    # Hyperparameters
    hp = yadll.hyperparameters.Hyperparameters()
    hp('batch_size', 500)
    hp('n_epochs', 1000)
    hp('learning_rate', 0.1)
    hp('momentum', 0.5)
    hp('l1_reg', 0.00)
    hp('l2_reg', 0.0000)
    hp('patience', 10000)
    # add the hyperparameters to the model
    model.hp = hp

We now create each layers of the network by implementing :class:`yadll.layers` classes.
We always start with a :class:`yadll.layers.Input` that give the shape of the input data.
This network will be a mlp with two dense layer with rectified linear unit activation and dropout.
Each layer receive as `incoming` the previous layer.
The last layer is a :class:`yadll.layers.LogisticRegression` which is a dense layer with softmax activation.
Layers names are optional.

.. code-block:: python

    # Create connected layers
    # Input layer
    l_in = yadll.layers.InputLayer(shape=(hp.batch_size, 28 * 28), name='Input')
    # Dropout Layer 1
    l_dro1 = yadll.layers.Dropout(incoming=l_in, corruption_level=0.4, name='Dropout 1')
    # Dense Layer 1
    l_hid1 = yadll.layers.DenseLayer(incoming=l_dro1, nb_units=500, W=yadll.init.glorot_uniform, l1=hp.l1_reg,
                                     l2=hp.l2_reg, activation=yadll.activation.relu, name='Hidden layer 1')
    # Dropout Layer 2
    l_dro2 = yadll.layers.Dropout(incoming=l_hid1, corruption_level=0.2, name='Dropout 2')
    # Dense Layer 2
    l_hid2 = yadll.layers.DenseLayer(incoming=l_dro2, nb_units=500, W=yadll.init.glorot_uniform, l1=hp.l1_reg,
                                     l2=hp.l2_reg, activation=yadll.activation.relu, name='Hidden layer 2')
    # Logistic regression Layer
    l_out = yadll.layers.LogisticRegression(incoming=l_hid2, nb_class=10, l1=hp.l1_reg,
                                            l2=hp.l2_reg, name='Logistic regression')

We create a :class:`yadll.network.Network` object and add all the layers sequentially.
Order matters!!!

.. code-block:: python

    # Create network and add layers
    net = yadll.network.Network('2 layers mlp with dropout')
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

    # updates method
    model.updates = yadll.updates.nesterov_momentum

    # train the model and save it to file at each best
    model.train(save_mode='each')

Here is the output when trained on NVIDIA Geforce Titan X card:

.. code-block:: text

    epoch 463, minibatch 100/100, validation error 1.360 %
    epoch 464, minibatch 100/100, validation error 1.410 %
    epoch 465, minibatch 100/100, validation error 1.400 %

     Optimization completed. Early stopped at epoch: 466
     Validation score of 1.260 % obtained at iteration 23300, with test performance 1.320 %
     Training mlp with dropout took 02 m 29 s


Making Prediction
_________________

Once the model is trained let's use it to make prediction:

.. code-block:: python

    # make prediction
    # We can test it on some examples from test
    test_set_x = data.test_set_x.get_value()
    test_set_y = data.test_set_y.eval()

    predicted_values = model.predict(test_set_x[:30])

    print ("Predicted values for the first 30 examples in test set:")
    print predicted_values
    print test_set_y[:30]


Saving/loading models
---------------------
Yadll provides two ways to save and load models.
The first will pickle the whole model. It is not recommended for long term
storage but is very convenient to handle models.
The second is more robust. It saves the parameters of the network but ask you to
recreate the network and model.


Run the mnist examples
----------------------

Different networks are tested on MNIST dataset in the ``examples/mnist_examples.py``
file.

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

  python mnist_examples.py --network_list


Trainning a model for example lenet5:

.. code-block:: bash

  python mnist_examples.py lenet5


Grid search of the Hyperparameters
----------------------------------

grid search on the hyperparameters:

.. code-block:: bash

  python hp_grid_search.py