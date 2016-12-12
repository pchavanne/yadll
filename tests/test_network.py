#!/usr/bin/env python
from mock import MagicMock as mock
import theano.tensor as T
import numpy as np
import pytest
import logging


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
        return InputLayer(input_shape=(None, 25), input=x)

    @pytest.fixture
    def layer(self, input):
        from yadll.layers import DenseLayer
        return DenseLayer(incoming=input, nb_units=25, name='DenseLayer 1')

    @pytest.fixture
    def layer2(self, input):
        from yadll.layers import DenseLayer
        return DenseLayer(incoming=input, nb_units=25)

    @pytest.fixture
    def unsupervised_layer(self, layer):
        from yadll.layers import AutoEncoder
        return AutoEncoder(incoming=layer, nb_units=25, hyperparameters=mock())

    def test_network(self, network, x, input, layer, layer2, unsupervised_layer, caplog):
        net = network(name='test_network', layers=[input, layer, unsupervised_layer])
        assert net.params == [layer.W, layer.b, unsupervised_layer.W, unsupervised_layer.b]
        net = network(name='test_network')
        net.add(input)
        net.add(layer)
        assert net.reguls == 0
        assert net.params == [layer.W, layer.b]
        assert net.name == 'test_network'
        assert net.get_layer('DenseLayer 1') is layer
        assert net['DenseLayer 1'] is layer
        assert net.has_unsupervised_layer is False
        net.add(unsupervised_layer)
        assert net.has_unsupervised_layer is True
        from yadll.utils import to_float_X
        x_val = to_float_X(np.random.random((50, 25)))
        assert (net.get_output().eval({x: x_val}) == unsupervised_layer.get_output().eval({x: x_val})).all()
        # Save and load Params
        net.save_params('test.yp')
        net2 = network(name='test_network', layers=[input, layer2, unsupervised_layer])

        def params_equal(params1, params2):
            for i in range(len(params1)):
                np.testing.assert_array_equal(params1[i].eval(),
                                              params2[i].eval())

        np.testing.assert_raises(AssertionError, params_equal,
                                 net.params, net2.params)
        net2.load_params('test.yp')
        params_equal(net.params, net2.params)

        caplog.setLevel(logging.ERROR)
        net3 = network(name=None)
        net3.add(input)
        net3.add(layer)
        net3.save_params('test.yp')
        assert 'Your network has no name. Please set one and try again.'in caplog.text()
        net3.load_params('test.yp')
        assert 'Network names are different. Saved network name is: test_network' in caplog.text()
