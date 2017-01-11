.. _tutorial:

========
Tutorial
========

Building and training your first network
========================================

Let's build our first MLP with dropout on the MNIST example.
to run this example, just do:

.. code-block:: bash

    cd /yadll/examples
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
name is provided, the model will be saved (see `Saving/loading models`_).

.. code-block:: python

    # create the model
    model = yadll.model.Model(name='mlp with dropout', data=data, file='best_model.ym')

We define the hyperparameters(see `Hyperparameters and Grid search`_) of the model and add it to our model object.

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
The first layers must be a :class:`yadll.layers.Input` that give the shape of the input data.
This network will be a mlp with two dense layer with rectified linear unit activation and dropout.
Each layer receive as `incoming` the previous layer.
Each layer has a name. You can provide it or it will be, by default, the name of the layer class, space, the number of
the instantiation.
The last layer is a :class:`yadll.layers.LogisticRegression` which is a dense layer with softmax activation.
Layers names are optional.

.. code-block:: python

    # Create connected layers
    # Input layer
    l_in = yadll.layers.InputLayer(shape=(hp.batch_size, 28 * 28), name='Input')
    # Dropout Layer 1
    l_dro1 = yadll.layers.Dropout(incoming=l_in, corruption_level=0.4, name='Dropout 1')
    # Dense Layer 1
    l_hid1 = yadll.layers.DenseLayer(incoming=l_dro1, nb_units=500, W=yadll.init.glorot_uniform,
                                     l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu,
                                     name='Hidden layer 1')
    # Dropout Layer 2
    l_dro2 = yadll.layers.Dropout(incoming=l_hid1, corruption_level=0.2, name='Dropout 2')
    # Dense Layer 2
    l_hid2 = yadll.layers.DenseLayer(incoming=l_dro2, nb_units=500, W=yadll.init.glorot_uniform,
                                     l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu,
                                     name='Hidden layer 2')
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
Here we update with the stochastic gradient descent with Nesterov momentum.

.. code-block:: python

    # add the network to the model
    model.network = net

    # updates method
    model.updates = yadll.updates.nesterov_momentum

    # train the model and save it to file at each best
    model.train(save_mode='each')

Here is the output when trained on a NVIDIA Geforce Titan X card:

.. code-block:: text

    epoch 463, minibatch 100/100, validation error 1.360 %
    epoch 464, minibatch 100/100, validation error 1.410 %
    epoch 465, minibatch 100/100, validation error 1.400 %

     Optimization completed. Early stopped at epoch: 466
     Validation score of 1.260 % obtained at iteration 23300, with test performance 1.320 %
     Training mlp with dropout took 02 m 29 s


Making Prediction
=================
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

This should give you

.. code-block:: text

    Predicted values for the first 30 examples in test set:
    [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4 9 6 6 5 4 0 7 4 0 1]
    [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4 9 6 6 5 4 0 7 4 0 1]


Saving/loading models
=====================

Yadll provides two ways to save and load models.

Save the model
^^^^^^^^^^^^^^

This first method for saving your model is to pickle the whole model. It is not recommended for long term
storage but is very convenient to handle models. All you have to do is provide you model constructor with
a file name. The model will be saved after training.

.. code-block:: python

    model = yadll.model.Model(name='mlp with dropout', data=data, file='best_model.ym')

You can also save your model by setting the `save_mode` argument of the train function.
If you didn't give a file name to the constructor it will create one (model.name + '_YmdHMS.ym').
You can set it to 'end' (save at the end of the training) or 'each' (save after each best model).

.. code-block:: python

    model.train(save_mode='each')

If you used 'each' and if your system crash you will be able to
restart the training from the last best model.

To load the model just do

.. code-block:: python

    # load the saved model
    model2 = yadll.model.load_model('best_model.ym')

.. warning::

    * Do not use this method for long term storage or production environment.
    * Model trained on GPU will not be usable on CPU.


Save the network parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This second method is more robust and can be used for long term storage.
It consists in saving the parameters (pickling) of the network.

Once the model has trained the network you can save its parameters

.. code-block:: python

    # saving network parameters
    net.save_params('net_params.yp')

Now you can retrieve the model with those parameters, but first you have to recreate the model.
When loading the parameters, the network name must match the saved parameters network name.

.. code-block:: python

    # load network parameters
    # first we recreate the network
    # create the model
    model3 = yadll.model.Model(name='mlp with dropout', data=data,)

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
    model3.hp = hp

    # Create connected layers
    # Input layer
    l_in = yadll.layers.InputLayer(shape=(hp.batch_size, 28 * 28), name='Input')
    # Dropout Layer 1
    l_dro1 = yadll.layers.Dropout(incoming=l_in, corruption_level=0.4, name='Dropout 1')
    # Dense Layer 1
    l_hid1 = yadll.layers.DenseLayer(incoming=l_dro1, nb_units=500, W=yadll.init.glorot_uniform,
                                     l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu,
                                     name='Hidden layer 1')
    # Dropout Layer 2
    l_dro2 = yadll.layers.Dropout(incoming=l_hid1, corruption_level=0.2, name='Dropout 2')
    # Dense Layer 2
    l_hid2 = yadll.layers.DenseLayer(incoming=l_dro2, nb_units=500, W=yadll.init.glorot_uniform,
                                     l1=hp.l1_reg, l2=hp.l2_reg, activation=yadll.activations.relu,
                                     name='Hidden layer 2')
    # Logistic regression Layer
    l_out = yadll.layers.LogisticRegression(incoming=l_hid2, nb_class=10, l1=hp.l1_reg,
                                            l2=hp.l2_reg, name='Logistic regression')

    # Create network and add layers
    net2 = yadll.network.Network('2 layers mlp with dropout')
    net2.add(l_in)
    net2.add(l_dro1)
    net2.add(l_hid1)
    net2.add(l_dro2)
    net2.add(l_hid2)
    net2.add(l_out)

    # load params
    net2.load_params('net_params.yp')   # Here we don't train the model but reload saved parameters

    # add the network to the model
    model3.network = net2

Save the configuration
^^^^^^^^^^^^^^^^^^^^^^

Models can be saved as configuration objects or files.

.. code-block:: python

    # Saving configuration of the model. Model doesn't have to be trained
    conf = model.to_conf()    # get the configuration
    model.to_conf('conf.yc')  # or save it to file .yc by convention


and reloaded:

.. code-block:: python

    # Reconstruction the model from configuration and load paramters
    model4 = yadll.model.Model()
    model4.from_conf(conf)         # load from conf obj
    model5 = yadll.model.Model()
    model5.from_conf(file='conf.yc')    # load from conf file

You can now reload parameters or train the network.

Networks can be modified directly from the conf object.

.. note::
    By convention we use the .ym extension for Yadll Model file,
    .yp for Yadll Parameters file and .yc for configuration but it is not mandatory.

Run the examples
================

Yadll provide a rather exhaustive list of conventional network implementation.
You will find them in the ``/yadll/examples/networks.py`` file.

MNIST
-----

Let's try those network on the MNIST dataset. in the ``/yadll/examples/mnist_examples.py``
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


Training a model for example lenet5:

.. code-block:: bash

  python mnist_examples.py lenet5


Hyperparameters and Grid search
===============================

Yadll provide the :class:`yadll.hyperparameters.Hyperparameters` to hold the
hyperparameters of the model. It also allows to perform a grid search optimisation
as the class is iterable over all hyperparameters combinations.

Let's first define our hyperparameters and their search space

.. code-block:: python

    # Hyperparameters
    hps = Hyperparameters()
    hps('batch_size', 500, [50, 100, 500, 1000])
    hps('n_epochs', 1000)
    hps('learning_rate', 0.1, [0.001, 0.01, 0.1, 1])
    hps('l1_reg', 0.00, [0, 0.0001, 0.001, 0.01])
    hps('l2_reg', 0.0001, [0, 0.0001, 0.001, 0.01])
    hps('activation', tanh, [tanh, sigmoid, relu])
    hps('initialisation', glorot_uniform, [glorot_uniform, glorot_normal])
    hps('patience', 10000)

Now we will loop over each possible combination

.. code-block:: python

    reports = []
    for hp in hps:
        # create the model
        model = Model(name='mlp grid search', data=data)
        # add the hyperparameters to the model
        model.hp = hp
        # Create connected layers
        # Input layer
        l_in = InputLayer(shape=(None, 28 * 28), name='Input')
        # Dense Layer 1
        l_hid1 = DenseLayer(incoming=l_in, nb_units=5, W=hp.initialisation, l1=hp.l1_reg,
                            l2=hp.l2_reg, activation=hp.activation, name='Hidden layer 1')
        # Dense Layer 2
        l_hid2 = DenseLayer(incoming=l_hid1, nb_units=5, W=hp.initialisation, l1=hp.l1_reg,
                            l2=hp.l2_reg, activation=hp.activation, name='Hidden layer 2')
        # Logistic regression Layer
        l_out = LogisticRegression(incoming=l_hid2, nb_class=10, l1=hp.l1_reg,
                                   l2=hp.l2_reg, name='Logistic regression')

        # Create network and add layers
        net = Network('mlp')
        net.add(l_in)
        net.add(l_hid1)
        net.add(l_hid2)
        net.add(l_out)
        # add the network to the model
        model.network = net

        # updates method
        model.updates = yadll.updates.sgd
        reports.append((hp, model.train()))

.. warning::
    These hyperparameters would generate 4*4*4*4*3*2=1536 different combinations.
    Each of these combinations would have a different training time but
    if it takes 10 minutes on average, the whole optimisation would last more the 10 days!!!


to run this example, just do:

.. code-block:: bash

    python hp_grid_search.py


