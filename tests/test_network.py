#!/usr/bin/env python
from mock import MagicMock as mock
import theano.tensor as T
import numpy as np
import pytest


class TestNetwork:
    @pytest.fixture
    def network(self):
        from yadll.network import Network
        return Network

    @pytest.fixture(scope='module')
    def x(self):
        from yadll.utils import floatX
        return T.matrix(name='x', dtype=floatX)

    @pytest.fixture
    def input(self, x):
        from yadll.layers import InputLayer
        return InputLayer(shape=(50, 25), input_var=x)

    @pytest.fixture
    def layer(self, input):
        from yadll.layers import DenseLayer
        return DenseLayer(incoming=input, nb_units=25)

    @pytest.fixture
    def unsupervised_layer(self, layer):
        from yadll.layers import AutoEncoder
        return AutoEncoder(incoming=layer, nb_units=25, hyperparameters=mock())

    def test_network(self, network, x, input, layer, unsupervised_layer):
        net = network(name='test_network', layers=[input, layer, unsupervised_layer])
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
        from yadll.utils import to_float_X
        x_val = to_float_X(np.random.random((50, 25)))
        assert (net.get_output().eval({x: x_val}) == unsupervised_layer.get_output().eval({x: x_val})).all()

