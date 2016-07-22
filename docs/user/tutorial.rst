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

**Save the model**

This first method for saving your model is to pickle the whole model. It is not recommended for long term
storage but is very convenient to handle models. All you have to do is provide you model constructor with
a file name. The model will be saved after training.

.. code-block:: python

    model = yadll.model.Model(name='mlp with dropout', data=data, file='best_model.ym')

You can also save your model by setting the `save_mode` argument of the train function.
If you didn't give a file name it will create one: model.name + '_' + ('%Y%m%d%H%M%S') + '.ym'.
You can set it to 'end' (save at the end of the training) or 'each' (save after each best model).

.. code-block:: python

    model.train(save_mode='each')

If you used 'each' and if your system crash you will be able to
restart the training from the last best model. To do so just do

To load the model just do

.. code-block:: python

    # load the saved model
    model2 = yadll.model.load_model('best_model.ym')

.. warning::

    * Do not use this method for long term storage or production environment.
    * Model trained on GPU will not be usable on CPU and vice versa.


**Save the network parameters**

This second method is more robust and can be used for long term storage.
It consists in saving the parameters (pickling)of the network. but ask you to
recreate the network and model.

Once the model has been trained you can save the parameters

.. code-block:: python

    # saving network paramters
    net.save_params('net_params.yp')

Now you can retrieve the model with those parameters, but you have to recreate the model and load the parameters.
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

.. note::
    By convention we use the .ym extension for Yadll Model file and
    .yp for Yadll Parameters file, but it is not mandatory.

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


Training a model for example lenet5:

.. code-block:: bash

  python mnist_examples.py lenet5


Grid search of the Hyperparameters
----------------------------------

grid search on the hyperparameters:

.. code-block:: bash

  python hp_grid_search.py

