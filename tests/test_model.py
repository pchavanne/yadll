# -*- coding: UTF-8 -*-
import numpy as np
import pytest

import logging

class TestModel:
    @pytest.fixture(scope='module')
    def data(self):
        from yadll.data import Data
        data = [[np.random.random((100, 25)), np.random.random_integers(low=0, high=9, size=(100,))],
                [np.random.random((50, 25)), np.random.random_integers(low=0, high=9, size=(50,))],
                [np.random.random((50, 25)), np.random.random_integers(low=0, high=9, size=(500,))]]
        return Data(data)

    @pytest.fixture(scope='module')
    def data_y_2D(self):
        from yadll.data import Data
        data = [[np.random.random((100, 25)), np.random.random_integers(low=0, high=9, size=(100,2))],
                [np.random.random((50, 25)), np.random.random_integers(low=0, high=9, size=(50,2))],
                [np.random.random((50, 25)), np.random.random_integers(low=0, high=9, size=(500,2))]]
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
    def model_y_2D(self, data_y_2D, hp):
        from yadll.model import Model
        return Model(name='test_model', data=data_y_2D, hyperparameters=hp)

    @pytest.fixture(scope='module')
    def input(self):
        from yadll.layers import InputLayer
        return InputLayer(input_shape=(None, 25))

    @pytest.fixture(scope='module')
    def layer(self, input):
        from yadll.layers import DenseLayer
        return DenseLayer(incoming=input, n_units=25, l1=0.1)

    @pytest.fixture(scope='module')
    def unsupervised_layer(self, layer):
        from yadll.layers import AutoEncoder
        from yadll.hyperparameters import Hyperparameters
        hp = Hyperparameters()
        hp('batch_size', 10)
        hp('n_epochs', 10)
        hp('learning_rate', 0.1)
        hp('patience', 1000)
        return AutoEncoder(incoming=layer, n_units=25, hyperparameters=hp)

    @pytest.fixture(scope='module')
    def logistic_regression_unsupervised(self, unsupervised_layer):
        from yadll.layers import LogisticRegression
        return LogisticRegression(incoming=unsupervised_layer, n_class=10)

    @pytest.fixture(scope='module')
    def logistic_regression(self, layer):
        from yadll.layers import LogisticRegression
        return LogisticRegression(incoming=layer, n_class=10)

    @pytest.fixture(scope='module')
    def network(self, input, layer, logistic_regression):
        from yadll.network import Network
        return Network(name='test_network', layers=[input, layer, logistic_regression])

    @pytest.fixture(scope='module')
    def network_unsupervised(self, input, unsupervised_layer, logistic_regression_unsupervised):
        from yadll.network import Network
        return Network(name='test_network', layers=[input, unsupervised_layer, logistic_regression_unsupervised])

    def test_no_data_found(self, model_no_data, network_unsupervised):
        model_no_data.network = network_unsupervised
        from yadll.exceptions import NoDataFoundException
        with pytest.raises(NoDataFoundException):
            model_no_data.pretrain()
        with pytest.raises(NoDataFoundException):
            model_no_data.train(unsupervised_training=False)

    def test_no_network(self, model):
        from yadll.exceptions import NoNetworkFoundException
        with pytest.raises(NoNetworkFoundException):
            model.pretrain()
        with pytest.raises(NoNetworkFoundException):
            model.train(unsupervised_training=False)

    def test_save_model(self, model, network, caplog):
        model.network = network
        from yadll.model import save_model, load_model, Model
        caplog.setLevel(logging.ERROR)
        save_model(model)
        assert 'No file name. Model not saved.' in caplog.text()
        model.train(save_mode='end')
        model.train(save_mode='each')
        model.train(save_mode='dummy')
        model.file=('test_model.ym')
        model.train()
        model.train(save_mode='end')
        model.train(save_mode='each')
        conf = model.to_conf()
        model.to_conf(file='test_conf.yc')
        save_model(model)
        save_model(model, 'test_model.ym')
        test_model = load_model('test_model.ym')
        model_from_conf = Model()
        model_from_conf.from_conf(conf)
        model_from_conf_file = Model()
        model_from_conf_file.from_conf(file='test_conf.yc')


    def test_model(self, model, model_y_2D, network, network_unsupervised):
        model.network = network
        assert model.name == 'test_model'
        model.train()
        network_unsupervised.layers[0].input = None
        model.network = network_unsupervised
        model.pretrain()
        model.train()
        model_y_2D.network = network
        #model_y_2D.train()

    def test_predict(self, data, model, network):
        model.network = network
        model.train()
        model.network.layers[0].input = None
        model.predict(data.test_set_x.eval()[:10])




