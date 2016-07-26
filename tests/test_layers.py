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
    def named_layer(self):
        from yadll.layers import Layer
        return Layer(mock(), name='layer_name')

    def test_input_shape(self, layer):
        assert layer.input_shape == layer.input_layer.output_shape

    def test_get_params(self, layer):
        assert layer.get_params() == []

    def test_get_reguls(self, layer):
        assert layer.get_reguls() == 0

    def test_named_layer(self, named_layer):
        assert named_layer.name == 'layer_name'

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


class TestInputLayer:
    @pytest.fixture
    def inputlayer(self):
        from yadll.layers import InputLayer
        return InputLayer((3, 2))

    def test_input_layer(self, inputlayer):
        assert inputlayer.input_layer is None

    def test_shape(self, inputlayer):
        assert inputlayer.input_shape == (3, 2)

    def test_get_params(self, inputlayer):
        assert inputlayer.get_params() == []

    def test_get_reguls(self, inputlayer):
        assert inputlayer.get_reguls() == 0


class TestReshapeLayer:
    @pytest.fixture
    def reshapelayer(self):
        from yadll.layers import ReshapeLayer
        return ReshapeLayer

    @pytest.fixture
    def inputdata(self):
        from yadll.utils import shared_variable
        return shared_variable(np.ones((16, 3, 5, 7, 10)))

    @pytest.fixture
    def inputlayer(self, inputdata):
        from yadll.layers import InputLayer
        shape = (16, 3, None, None, 10)
        return InputLayer(shape, input_var=inputdata)

    def test_reshape(self, reshapelayer, inputlayer):
        layer = reshapelayer(inputlayer, (16, 3, 5, 7, 2, 5))
        assert layer.output_shape == (16, 3, 5, 7, 2, 5)
        result = layer.get_output().eval()
        assert result.shape == (16, 3, 5, 7, 2, 5)


class TestFlattenLayer:
    @pytest.fixture
    def flattenlayer(self):
        from yadll.layers import FlattenLayer
        return FlattenLayer

    @pytest.fixture
    def inputdata(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((2, 3, 4, 5)))

    @pytest.fixture
    def inputlayer(self, inputdata):
        from yadll.layers import InputLayer
        shape = (2, 3, 4, 5,)
        return InputLayer(shape, input_var=inputdata)

    def test_output_shape(self, flattenlayer, inputlayer):
        layer = flattenlayer(inputlayer)
        assert layer.output_shape == (2, 3 * 4 * 5)

    def test_get_output(self, flattenlayer, inputlayer, inputdata):
        layer = flattenlayer(inputlayer)
        result = layer.get_output().eval()
        input = np.asarray(inputdata.eval())
        assert (result == input.reshape(input.shape[0], -1)).all()


class TestDenseLayer:
    @pytest.fixture
    def denselayer(self):
        from yadll.layers import DenseLayer
        return DenseLayer

    @pytest.fixture
    def inputdata(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20)))

    @pytest.fixture
    def inputlayer(self, inputdata):
        from yadll.layers import InputLayer
        shape = (10, 20)
        return InputLayer(shape, input_var=inputdata)

    @pytest.fixture
    def layer(self, denselayer, inputlayer):
        return denselayer(inputlayer, nb_units=2, l1=1, l2=2)

    def test_get_params(self, layer):
        assert layer.get_params() == [layer.W, layer.b]

    def test_output_shape(self, layer):
        assert layer.output_shape == (10, 2)

    def test_get_output(self, layer, inputdata):
        X = inputdata.eval()
        W = layer.W.eval()
        b = layer.b.eval()
        assert_allclose(layer.get_output().eval(), np.tanh(np.dot(X, W) + b), rtol=1e-4)

    def test_reguls(self, layer):
        W = layer.W.eval()
        assert_allclose(layer.reguls.eval(), np.mean(np.abs(W)) + 2 * np.mean(np.power(W, 2)), rtol=1e-4)


class TestUnsupervisedLayer:
    @pytest.fixture
    def unsupervisedlayer(self):
        from yadll.layers import UnsupervisedLayer
        return UnsupervisedLayer

    @pytest.fixture
    def inputdata(self):
        from yadll.utils import shared_variable
        return shared_variable(np.random.random((10, 20)))

    @pytest.fixture
    def inputlayer(self, inputdata):
        from yadll.layers import InputLayer
        shape = (10, 20)
        return InputLayer(shape, input_var=inputdata)

    @pytest.fixture
    def hp(self):
        from yadll.hyperparameters import Hyperparameters
        hp = Hyperparameters()
        hp('batch_size', 10)
        hp('n_epochs', 10)
        hp('learning_rate', 0.1)
        hp('patience', 1000)
        return hp

    @pytest.fixture
    def layer(self, unsupervisedlayer, inputlayer, hp):
        return unsupervisedlayer(inputlayer, nb_units=2, hyperparameters=hp)

    def test_get_params(self, layer):
        assert layer.get_params() == [layer.W, layer.b]

    def test_output_shape(self, layer):
        assert layer.output_shape == (10, 2)

    def test_get_output(self, layer, inputdata):
        X = inputdata.eval()
        W = layer.W.eval()
        b = layer.b.eval()
        assert_allclose(layer.get_output().eval(), np.tanh(np.dot(X, W) + b), rtol=1e-3)


class TestLogisticRegression:
    @pytest.fixture
    def logisticregression(self):
        from yadll.layers import LogisticRegression
        return LogisticRegression


class TestDropout:
    @pytest.fixture
    def dropout(self):
        from yadll.layers import Dropout
        return Dropout


class TestDropConnect:
    @pytest.fixture
    def dropout(self):
        from yadll.layers import Dropconnect
        return Dropconnect


class TestDropPoolLayer:
    @pytest.fixture
    def dropout(self):
        from yadll.layers import PoolLayer
        return PoolLayer


class TestConvLayer:
    @pytest.fixture
    def dropout(self):
        from yadll.layers import ConvLayer
        return ConvLayer


class TestConvPoolLayer:
    @pytest.fixture
    def dropout(self):
        from yadll.layers import ConvPoolLayer
        return ConvPoolLayer


class TestAutoEncoder:
    @pytest.fixture
    def dropout(self):
        from yadll.layers import AutoEncoder
        return AutoEncoder


class TestRBM:
    @pytest.fixture
    def dropout(self):
        from yadll.layers import RBM
        return RBM


class TestRNN:
    @pytest.fixture
    def dropout(self):
        from yadll.layers import RNN
        return RNN


class TestLSTM:
    @pytest.fixture
    def dropout(self):
        from yadll.layers import LSTM
        return LSTM