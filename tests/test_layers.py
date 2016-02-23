# -*- coding: UTF-8 -*-
import pytest
from mock import Mock
import numpy as np


class TestLayer:
    @pytest.fixture
    def layer(self):
        from dl.layers import Layer
        return Layer(Mock())

    @pytest.fixture
    def named_layer(self):
        from dl.layers import Layer
        return Layer(Mock(), name='layer_name')

    def test_input_shape(self, layer):
        assert layer.input_shape == layer.input_layer.output_shape

    def test_get_params(self, layer):
        assert layer.get_params() == []

    def test_get_reguls(self, layer):
        assert layer.get_reguls() == 0

    def test_named_layer(self, named_layer):
        assert named_layer.name == 'layer_name'

    @pytest.fixture
    def layer_from_shape(self):
        from dl.layers import Layer
        return Layer((None, 20))

    def test_layer_from_shape(self, layer_from_shape):
        layer = layer_from_shape
        assert layer.input_layer is None
        assert layer.input_shape == (None, 20)


class TestInputLayer:
    @pytest.fixture
    def inputlayer(self):
        from dl.layers import InputLayer
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
        from dl.layers import ReshapeLayer
        return ReshapeLayer

    @pytest.fixture
    def inputdata(self):
        from dl.utils import shared_variable
        return shared_variable(np.ones((16, 3, 5, 7, 10)))

    @pytest.fixture
    def inputlayer(self, inputdata):
        from dl.layers import InputLayer
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
        from dl.layers import FlattenLayer
        return FlattenLayer

    @pytest.fixture
    def inputdata(self):
        from dl.utils import shared_variable
        return shared_variable(np.random.random((2, 3, 4, 5)))

    @pytest.fixture
    def inputlayer(self, inputdata):
        from dl.layers import InputLayer
        shape = (2, 3, 4, 5,)
        return InputLayer(shape, input_var=inputdata)

    def test_output_shape(self, flattenlayer, inputlayer):
        layer = flattenlayer(inputlayer)
        assert layer.output_shape == (2, 3 * 4 * 5)

    def test_get_output(self, flattenlayer, inputlayer, inputdata):
        layer = flattenlayer(inputlayer)
        result = layer.get_output().eval()
        input = inputdata.eval()
        assert (result == input.reshape(input.shape[0], -1)).all()


class TestDenseLayer:
    @pytest.fixture
    def denselayer(self):
        from dl.layers import DenseLayer
        return DenseLayer

    @pytest.fixture
    def inputdata(self):
        from dl.utils import shared_variable
        return shared_variable(np.random.random((10, 20)))

    @pytest.fixture
    def inputlayer(self, inputdata):
        from dl.layers import InputLayer
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
        assert (layer.get_output().eval() == np.tanh(np.dot(X, layer.W.eval()) + layer.b.eval())).all()

    def test_reguls(selfself, layer):
        assert layer.reguls.eval() == np.asarray(np.mean(np.abs(layer.W.eval())) + 2 * np.mean(layer.W.eval()**2), dtype='float32')
