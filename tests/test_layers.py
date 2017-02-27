# -*- coding: UTF-8 -*-
import pytest
from mock import MagicMock as mock
import numpy as np
from numpy.testing import assert_allclose


class TestLayer:
    @pytest.fixture
    def layer(self):
        from yadll.layers import Layer
        return Layer(mock())

    @pytest.fixture
    def layer2(self):
        from yadll.layers import Layer
        return Layer(mock())

    @pytest.fixture
    def named_layer(self):
        from yadll.layers import Layer
        return Layer(mock(), name='layer_name')

    def test_layer_from_none(self):
        from yadll.layers import Layer
        layer = Layer(incoming=None)
        assert layer.input_layer is None
        assert layer.input_shape is None

    def test_input_shape(self, layer):
        assert layer.input_shape == layer.input_layer.output_shape

    def test_get_params(self, layer):
        assert layer.get_params() == []

    def test_get_reguls(self, layer):
        assert layer.get_reguls() == 0

    def test_named_layer(self, named_layer):
        assert named_layer.name == 'layer_name'

    def test_unnamed_layer(self, layer):
        from yadll.layers import Layer
        assert layer.name == 'Layer ' + str(Layer.n_instances)

    def test_get_output(self, layer):
        with pytest.raises(NotImplementedError):
            layer.get_output()

    @pytest.fixture
    def layer_from_shape(self):
        from yadll.layers import Layer
        return Layer((None, 20))

    def test_layer_from_shape(self, layer_from_shape):
        layer = layer_from_shape
        assert layer.input_layer is None
        assert layer.input_shape == (None, 20)

    def test_layer_from_layers(self):
        from yadll.layers import Layer
        l_1 = Layer((10, 20))
        l_2 = Layer((30, 40))
        l = Layer([l_1, l_2])
        assert l.input_shape == [l_1.input_shape, l_2.input_shape]
        assert l.input_layer == [l_1, l_2]


class Testinput_layer:
    @pytest.fixture
    def input_layer(self):
        from yadll.layers import InputLayer
        return InputLayer((3, 2))

    def test_input_layer(self, input_layer):
        assert input_layer.input_layer is None

    def test_shape(self, input_layer):
        assert input_layer.input_shape == (3, 2)

    def test_get_params(self, input_layer):
        assert input_layer.get_params() == []

    def test_get_reguls(self, input_layer):
        assert input_layer.get_reguls() == 0


class TestReshapeLayer:
    @pytest.fixture
    def reshape_layer(self):
        from yadll.layers import ReshapeLayer
        return ReshapeLayer

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.ones((16, 3, 5, 7, 10)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (16, 3, None, None, 10)
        return InputLayer(shape, input=input_data)

    def test_reshape(self, reshape_layer, input_layer):
        layer = reshape_layer(input_layer, (16, 3, 5, 7, 2, 5))
        assert layer.output_shape == (16, 3, 5, 7, 2, 5)
        result = layer.get_output().eval()
        assert result.shape == (16, 3, 5, 7, 2, 5)
        layer = reshape_layer(input_layer, (None, 3, 5, 7, 2, 5))
        assert layer.output_shape == (None, 3, 5, 7, 2, 5)
        result = layer.get_output().eval()
        assert result.shape == (16, 3, 5, 7, 2, 5)


class TestFlattenLayer:
    @pytest.fixture
    def flatten_layer(self):
        from yadll.layers import FlattenLayer
        return FlattenLayer

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((2, 3, 4, 5)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (2, 3, 4, 5,)
        return InputLayer(shape, input=input_data)

    def test_output_shape(self, flatten_layer, input_layer):
        layer = flatten_layer(incoming=input_layer)
        assert layer.output_shape == (2, 3 * 4 * 5)

    def test_get_output(self, flatten_layer, input_layer, input_data):
        layer = flatten_layer(incoming=input_layer)
        result = layer.get_output().eval()
        input = np.asarray(input_data.eval())
        assert (result == input.reshape(input.shape[0], -1)).all()


class TestActivation:
    @pytest.fixture
    def activation(self):
        from yadll.layers import Activation
        return Activation

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((2, 3, 4, 5)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (2, 3, 4, 5,)
        return InputLayer(shape, input=input_data)

    def test_output_shape(self, activation, input_layer):
        layer = activation(incoming=input_layer)
        assert layer.output_shape == (2, 3 * 4 * 5)

    def test_get_output(self, activation, input_layer, input_data):
        from yadll.activations import tanh
        layer = activation(input_layer, activation=tanh)
        result = layer.get_output().eval()
        input = np.asarray(input_data.eval())
        assert np.allclose(result, np.tanh(input))


class TestDenseLayer:
    @pytest.fixture
    def dense_layer(self):
        from yadll.layers import DenseLayer
        return DenseLayer

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (10, 20)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def layer(self, dense_layer, input_layer):
        return dense_layer(incoming=input_layer, n_units=2, l1=1, l2=2)

    @pytest.fixture
    def layer_from_layer(self, dense_layer, input_layer, layer):
        return dense_layer(incoming=input_layer, W=layer.W, b=layer.b, n_units=2, l1=1, l2=2)

    def test_get_params(self, layer):
        assert layer.get_params() == [layer.W, layer.b]

    def test_output_shape(self, layer):
        assert layer.output_shape == (10, 2)

    def test_get_output(self, layer, input_data):
        X = input_data.eval()
        W = layer.W.eval()
        b = layer.b.eval()
        assert_allclose(layer.get_output().eval(), np.tanh(np.dot(X, W) + b), rtol=1e-4)

    def test_reguls(self, layer):
        W = layer.W.eval()
        assert_allclose(layer.reguls.eval(), np.mean(np.abs(W)) + 2 * np.mean(np.power(W, 2)), rtol=1e-4)

    def test_layer_from_layer(self, layer, layer_from_layer):
        assert layer.W == layer_from_layer.W
        assert layer.b == layer_from_layer.b


class Testunsupervised_layer:
    @pytest.fixture
    def unsupervised_layer(self):
        from yadll.layers import UnsupervisedLayer
        return UnsupervisedLayer

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (10, 20)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def hp(self):
        from yadll.hyperparameters import Hyperparameters
        hp = Hyperparameters()
        hp('batch_size', 10)
        hp('n_epochs', 10)
        hp('learning_rate', 0.1)
        hp('patience', 100)
        return hp

    @pytest.fixture
    def layer(self, unsupervised_layer, input_layer, hp):
        return unsupervised_layer(incoming=input_layer, n_units=2, hyperparameters=hp)

    def test_get_params(self, layer):
        assert layer.get_params() == [layer.W, layer.b]

    def test_output_shape(self, layer):
        assert layer.output_shape == (10, 2)

    def test_get_output(self, layer, input_data):
        X = input_data.eval()
        W = layer.W.eval()
        b = layer.b.eval()
        assert_allclose(layer.get_output().eval(), np.tanh(np.dot(X, W) + b), rtol=1e-3)

    def test_get_encoded_input(self, layer):
        with pytest.raises(NotImplementedError):
            layer.get_encoded_input()

    def test_get_unsupervised_cost(self, layer):
        with pytest.raises(NotImplementedError):
            layer.get_unsupervised_cost()


class TestLogisticRegression:
    @pytest.fixture
    def logistic_regression(self):
        from yadll.layers import LogisticRegression
        return LogisticRegression


class TestDropout:
    @pytest.fixture
    def dropout(self):
        from yadll.layers import Dropout
        return Dropout

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (None, 20)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def layer(self, dropout, input_layer):
        return dropout(incoming=input_layer, corruption_level=0.5)

    @pytest.fixture
    def layer_c0(self, dropout, input_layer):
        return dropout(incoming=input_layer, corruption_level=0)

    @pytest.fixture
    def layer_c1(self, dropout, input_layer):
        return dropout(incoming=input_layer, corruption_level=1)

    def test_get_output(self, input_layer, layer_c0, layer_c1):
        np.testing.assert_array_equal(input_layer.get_output().eval(), layer_c0.get_output().eval())
        assert np.all(layer_c1.get_output().eval() == 0)


class TestDropConnect:
    @pytest.fixture
    def dropconnect(self):
        from yadll.layers import Dropconnect
        return Dropconnect

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (10, 20)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def layer(self, dropconnect, input_layer):
        return dropconnect(incoming=input_layer, n_units=10, corruption_level=0.5)

    @pytest.fixture
    def layer_c0(self, dropconnect, input_layer):
        return dropconnect(incoming=input_layer, n_units=10, corruption_level=0)

    @pytest.fixture
    def layer_c1(self, dropconnect, input_layer):
        return dropconnect(incoming=input_layer, n_units=10, corruption_level=1)

    def test_get_output(self, layer_c1):
        assert np.all(layer_c1.get_output().eval() == 0)


class TestPoolLayer:
    @pytest.fixture
    def pool_layer(self):
        from yadll.layers import PoolLayer
        return PoolLayer

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (10, 20)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def layer(self, pool_layer, input_layer):
        return pool_layer(incoming=input_layer, pool_size=(2, 2))

    def test_get_output(self, layer):
        output = layer.get_output().eval()


class TestConvLayer:
    @pytest.fixture
    def conv_layer(self):
        from yadll.layers import ConvLayer
        return ConvLayer

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((100, 1, 28, 28)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (None, 1, 28, 28)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def layer(self, conv_layer, input_layer):
        return conv_layer(incoming=input_layer, image_shape=(None, 1, 28, 28), filter_shape=(20, 1, 5, 5))

    def test_get_output(self, layer):
        output = layer.get_output().eval()


class TestConvPoolLayer:
    @pytest.fixture
    def conv_pool_layer(self):
        from yadll.layers import ConvPoolLayer
        return ConvPoolLayer

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((100, 1, 28, 28)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (None, 1, 28, 28)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def layer(self, conv_pool_layer, input_layer):
        return conv_pool_layer(incoming=input_layer, image_shape=(None, 1, 28, 28), filter_shape=(20, 1, 5, 5), pool_size=(2, 2))

    def test_get_output(self, layer):
        output = layer.get_output().eval()


class TestAutoEncoder:
    @pytest.fixture
    def auto_encoder(self):
        from yadll.layers import AutoEncoder
        return AutoEncoder

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (None, 20)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def unsupervised_hp(self):
        from yadll.hyperparameters import Hyperparameters
        hp = Hyperparameters()
        hp('batch_size', 10)
        hp('n_epochs', 15)
        hp('learning_rate', 0.01)
        return hp

    @pytest.fixture
    def layer(self, auto_encoder, input_layer, unsupervised_hp):
        return auto_encoder(incoming=input_layer, n_units=10, hyperparameters=unsupervised_hp)

    def test_get_output(self, layer):
        output = layer.get_output().eval()


class TestRBM:
    @pytest.fixture
    def rbm(self):
        from yadll.layers import RBM
        return RBM

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (None, 20)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def unsupervised_hp(self):
        from yadll.hyperparameters import Hyperparameters
        hp = Hyperparameters()
        hp('batch_size', 10)
        hp('n_epochs', 15)
        hp('learning_rate', 0.01)
        return hp

    @pytest.fixture
    def layer(self, rbm, input_layer, unsupervised_hp):
        return rbm(incoming=input_layer, n_units=10, hyperparameters=unsupervised_hp)

    def test_get_output(self, layer):
        output = layer.get_output().eval()


class TestBatchNormalization:
    @pytest.fixture
    def batch_normalization(self):
        from yadll.layers import BatchNormalization
        return BatchNormalization

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (10, 20)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def layer(self, batch_normalization, input_layer):
        return batch_normalization(incoming=input_layer)

    def test_get_output(self, layer):
        output = layer.get_output().eval()


class TestRNN:
    @pytest.fixture
    def rnn(self):
        from yadll.layers import RNN
        return RNN

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20, 30)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (None, 20, 30)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def layer(self, rnn, input_layer):
        return rnn(incoming=input_layer, n_units=100)

    def test_get_output(self, layer):
        output = layer.get_output().eval()


class TestLSTM:
    @pytest.fixture
    def lstm(self):
        from yadll.layers import LSTM
        return LSTM

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20, 30)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (None, 20, 30)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def layer(self, lstm, input_layer):
        return lstm(incoming=input_layer, n_units=10)

    @pytest.fixture
    def peepholes_layer(self, lstm, input_layer):
        return lstm(incoming=input_layer, n_units=10, peepholes=True)

    @pytest.fixture
    def tied_layer(self, lstm, input_layer):
        return lstm(incoming=input_layer, n_units=10, tied_i_f=True)

    def test_get_output(self, layer, peepholes_layer, tied_layer):
        output = layer.get_output().eval()
        output = peepholes_layer.get_output().eval()
        output = tied_layer.get_output().eval()


class TestGRU:
    @pytest.fixture
    def gru(self):
        from yadll.layers import GRU
        return GRU

    @pytest.fixture
    def input_data(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20, 30)))

    @pytest.fixture
    def input_layer(self, input_data):
        from yadll.layers import InputLayer
        shape = (None, 20, 30)
        return InputLayer(shape, input=input_data)

    @pytest.fixture
    def layer(self, gru, input_layer):
        return gru(incoming=input_layer, n_units=10)

    def test_get_output(self, layer):
        output = layer.get_output().eval()