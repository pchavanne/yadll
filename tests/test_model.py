# -*- coding: UTF-8 -*-
from mock import MagicMock as mock
import theano.tensor as T
import numpy as np
import pytest


class TestNetwork:
    @pytest.fixture
    def network(self):
        from dl.model import Network
        return Network

    @pytest.fixture(scope='module')
    def x(self):
        from dl.utils import floatX
        return T.matrix(name='x', dtype=floatX)

    @pytest.fixture
    def input(self, x):
        from dl.layers import InputLayer
        return InputLayer(shape=(50, 25), input_var=x)

    @pytest.fixture
    def layer(self, input):
        from dl.layers import DenseLayer
        return DenseLayer(incoming=input, nb_units=25)

    @pytest.fixture
    def unsupervised_layer(self, layer):
        from dl.layers import AutoEncoder
        return AutoEncoder(incoming=layer, nb_units=25, hyperparameters=mock())

    def test_network(self, network, x, input, layer, unsupervised_layer):
        net = network(name='tes_network', layers=[input, layer, unsupervised_layer])
        assert net.params == [layer.W, layer.b, unsupervised_layer.W, unsupervised_layer.b]
        net = network(name='test_network')
        net.add(input)
        net.add(layer)
        assert net.reguls == 0
        assert net.params == [layer.W, layer.b]
        assert net.name == 'test_network'
        assert net.has_unsupervised_layer is False
        net.add(unsupervised_layer)
        assert net.has_unsupervised_layer is True
        from dl.utils import to_float_X
        x_val = to_float_X(np.random.random((50, 25)))
        assert (net.get_output().eval({x: x_val}) == unsupervised_layer.get_output().eval({x: x_val})).all()


class TestModel:
    @pytest.fixture(scope='module')
    def data(self):
        from dl.data import Data
        data = [[np.random.random((100, 25)), np.random.random_integers(low=0, high=9, size=(100,))],
                [np.random.random((50, 25)), np.random.random_integers(low=0, high=9, size=(50,))],
                [np.random.random((50, 25)), np.random.random_integers(low=0, high=9, size=(500,))]]
        return Data(data)

    @pytest.fixture(scope='module')
    def hp(self):
        from dl.hyperparameters import Hyperparameters
        hp = Hyperparameters()
        hp('batch_size', 10)
        hp('n_epochs', 105)
        hp('learning_rate', 0.1)
        hp('momentum', 0.9)
        hp('epsilon', 1e-6)
        hp('rho', 0.95)
        hp('l1_reg', 0.00)
        hp('l2_reg', 0.000)
        hp('patience', 1000)
        return hp

    @pytest.fixture(scope='module')
    def model(self, data, hp):
        from dl.model import Model
        return Model(name='test_model', data=data, hyperparameters=hp)

    @pytest.fixture(scope='module')
    def input(self, model, hp):
        from dl.layers import InputLayer
        return InputLayer(shape=(hp.batch_size, 25), input_var=model.x)

    @pytest.fixture(scope='module')
    def layer(self, input):
        from dl.layers import DenseLayer
        return DenseLayer(incoming=input, nb_units=25, l1=0.1)

    @pytest.fixture(scope='module')
    def unsupervised_layer(self, layer):
        from dl.layers import AutoEncoder
        from dl.hyperparameters import Hyperparameters
        hp = Hyperparameters()
        hp('batch_size', 10)
        hp('n_epochs', 10)
        hp('learning_rate', 0.1)
        hp('patience', 1000)
        return AutoEncoder(incoming=layer, nb_units=25, hyperparameters=hp)

    @pytest.fixture(scope='module')
    def logistic_regression(self, unsupervised_layer):
        from dl.layers import LogisticRegression
        return LogisticRegression(incoming=unsupervised_layer, nb_class=10)

    @pytest.fixture(scope='module')
    def network(self, input, layer, unsupervised_layer, logistic_regression):
        from dl.model import Network
        return Network(name='test_network', layers=[input, layer, unsupervised_layer, logistic_regression])

    def test_model(self, model, network):
        model.network = network
        assert model.name == 'test_model'
        model.train()




