# -*- coding: UTF-8 -*-
import numpy as np
import pytest


class TestModel:
    @pytest.fixture(scope='module')
    def data(self):
        from yadll.data import Data
        data = [[np.random.random((100, 25)), np.random.random_integers(low=0, high=9, size=(100,))],
                [np.random.random((50, 25)), np.random.random_integers(low=0, high=9, size=(50,))],
                [np.random.random((50, 25)), np.random.random_integers(low=0, high=9, size=(500,))]]
        return Data(data)

    @pytest.fixture(scope='module')
    def hp(self):
        from yadll.hyperparameters import Hyperparameters
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
        from yadll.model import Model
        return Model(name='test_model', data=data, hyperparameters=hp)

    @pytest.fixture(scope='module')
    def model_no_data(self, hp):
        from yadll.model import Model
        return Model(name='test_model', hyperparameters=hp)

    @pytest.fixture(scope='module')
    def input(self):
        from yadll.layers import InputLayer
        return InputLayer(shape=(None, 25))

    @pytest.fixture(scope='module')
    def layer(self, input):
        from yadll.layers import DenseLayer
        return DenseLayer(incoming=input, nb_units=25, l1=0.1)

    @pytest.fixture(scope='module')
    def unsupervised_layer(self, layer):
        from yadll.layers import AutoEncoder
        from yadll.hyperparameters import Hyperparameters
        hp = Hyperparameters()
        hp('batch_size', 10)
        hp('n_epochs', 10)
        hp('learning_rate', 0.1)
        hp('patience', 1000)
        return AutoEncoder(incoming=layer, nb_units=25, hyperparameters=hp)

    @pytest.fixture(scope='module')
    def logistic_regression(self, unsupervised_layer):
        from yadll.layers import LogisticRegression
        return LogisticRegression(incoming=unsupervised_layer, nb_class=10)

    @pytest.fixture(scope='module')
    def network(self, input, layer, logistic_regression):
        from yadll.network import Network
        return Network(name='test_network', layers=[input, layer, logistic_regression])

    @pytest.fixture(scope='module')
    def network_unsupervised(self, input,unsupervised_layer, logistic_regression):
        from yadll.network import Network
        return Network(name='test_network', layers=[input, unsupervised_layer, logistic_regression])

    def test_no_data_found(self, model_no_data, network):
        model_no_data.network = network
        from yadll.exceptions import NoDataFoundException
        with pytest.raises(NoDataFoundException):
            model_no_data.pretrain()
        with pytest.raises(NoDataFoundException):
            model_no_data.train()

    def test_no_network(self, model):
        from yadll.exceptions import NoNetworkFoundException
        with pytest.raises(NoNetworkFoundException):
            model.pretrain()
        with pytest.raises(NoNetworkFoundException):
            model.train()

    def test_save_model(self, model, network):
        model.network = network
        from yadll.model import save_model, load_model
        model.train(save_mode='end')
        model.train(save_mode='each')
        model.file=('test_model.ym')
        model.train()
        model.train(save_mode='end')
        model.train(save_mode='each')
        save_model(model)
        save_model(model, 'test_model.ym')
        test_model = load_model('test_model.ym')

    def test_model(self, model, network, network_unsupervised):
        model.network = network
        assert model.name == 'test_model'
        model.train()
        model.network = network_unsupervised
        model.pretrain()




