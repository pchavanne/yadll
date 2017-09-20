# -*- coding: UTF-8 -*-
import pickle
import gzip
import pytest
import numpy as np
from numpy.testing import assert_allclose


def test_normalize():
    from yadll.data import normalize, apply_normalize, revert_normalize
    x = np.asarray([0, 1, 2, 3], dtype=float)
    z, min, max = normalize(x)
    assert_allclose(z, np.asarray([0., 0.33333333, 0.66666666, 1.]))
    assert min == 0
    assert max == 3
    assert apply_normalize(1.5, min, max) == 0.5
    assert revert_normalize(0.5, min, max) == 1.5


def test_standardize():
    from yadll.data import standardize, apply_standardize, revert_standardize
    x = np.asarray([0, 1, 2, 3], dtype=float)
    z, mean, std = standardize(x)
    assert_allclose(z, np.asarray([-1.34163959, -0.4472132, 0.4472132, 1.34163959]))
    assert mean == 1.5
    assert std == 1.1180349887498948
    assert apply_standardize(1.5, mean, std) == 0.0
    assert revert_standardize(0.0, mean, std) == 1.5
    assert apply_standardize(2.618034988749895, mean, std) - 1 < 1e-6
    assert revert_standardize(1.0, mean, std) - 2.618034988749895 < 1e-6

def test_one_hot_encoding():
    from yadll.data import one_hot_encoding
    np.testing.assert_array_equal(one_hot_encoding(np.asarray([1, 0, 3])),
                                  np.asarray([[0, 1, 0, 0],
                                              [1, 0, 0, 0],
                                              [0, 0, 0, 1]]))
    np.testing.assert_array_equal(one_hot_encoding(np.asarray([1, 0, 3]), 5),
                                  np.asarray([[0, 1, 0, 0, 0, 0],
                                              [1, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0]]))


def test_one_hot_decoding():
    from yadll.data import one_hot_decoding
    np.testing.assert_array_equal(one_hot_decoding(np.asarray([[0, 1, 0, 0],
                                                               [1, 0, 0, 0],
                                                               [0, 0, 0, 1]])),
                                  np.asarray([1, 0, 3]))


def test_mnist_loader():
    from yadll.data import mnist_loader
    data = mnist_loader()


def test_alphabet_loader():
    from yadll.data import alphabet_loader
    data = alphabet_loader(2)


class Testdata:
    @pytest.fixture
    def train_valid_test_data(self):
        return [[1., 2.], [3., 4.], [5., 6.]]

    @pytest.fixture
    def train_test_data(self):
        return [[1., 2.], [5., 6.]]

    def test_train_valid_test_data(self, train_valid_test_data):
        from yadll.data import Data
        data = Data(train_valid_test_data, cast_y=False)
        assert np.asarray(data.train_set_x.eval()) == 1.
        assert np.asarray(data.train_set_y.eval()) == 2.
        assert np.asarray(data.valid_set_x.eval()) == 3.
        assert np.asarray(data.valid_set_y.eval()) == 4.
        assert np.asarray(data.test_set_x.eval()) == 5.
        assert np.asarray(data.test_set_y.eval()) == 6.
        assert data.train_set_y.eval().dtype == 'float32'
        assert data.valid_set_y.eval().dtype == 'float32'
        assert data.test_set_y.eval().dtype == 'float32'
        assert data.train_set_x.name == 'train_set_x'
        assert data.train_set_y.name == 'train_set_y'
        assert data.valid_set_x.name == 'valid_set_x'
        assert data.valid_set_y.name == 'valid_set_y'
        assert data.test_set_x.name == 'test_set_x'
        assert data.test_set_y.name == 'test_set_y'
        data = Data(train_valid_test_data, cast_y=True)
        assert np.asarray(data.train_set_x.eval()) == 1.
        assert np.asarray(data.train_set_y.eval()) == 2.
        assert np.asarray(data.valid_set_x.eval()) == 3.
        assert np.asarray(data.valid_set_y.eval()) == 4.
        assert np.asarray(data.test_set_x.eval()) == 5.
        assert np.asarray(data.test_set_y.eval()) == 6.
        assert data.train_set_y.eval().dtype == 'int32'
        assert data.valid_set_y.eval().dtype == 'int32'
        assert data.test_set_y.eval().dtype == 'int32'
        assert data.train_set_x.name == 'train_set_x'
        assert data.valid_set_x.name == 'valid_set_x'
        assert data.test_set_x.name == 'test_set_x'
        assert np.asarray(data.dataset()[0][0].eval()) == 1
        assert np.asarray(data.dataset()[0][1].eval()) == 2
        assert np.asarray(data.dataset()[1][0].eval()) == 3
        assert np.asarray(data.dataset()[1][1].eval()) == 4
        assert np.asarray(data.dataset()[2][0].eval()) == 5
        assert np.asarray(data.dataset()[2][1].eval()) == 6

    def test_train_test_data(self, train_test_data):
        from yadll.data import Data
        data = Data(train_test_data, cast_y=False)
        assert np.asarray(data.train_set_x.eval()) == 1.
        assert np.asarray(data.train_set_y.eval()) == 2.
        assert data.valid_set_x is None
        assert data.valid_set_y is None
        assert np.asarray(data.test_set_x.eval()) == 5.
        assert np.asarray(data.test_set_y.eval()) == 6.
        assert data.train_set_y.eval().dtype == 'float32'
        assert data.test_set_y.eval().dtype == 'float32'
        assert data.train_set_x.name == 'train_set_x'
        assert data.train_set_y.name == 'train_set_y'
        assert data.test_set_x.name == 'test_set_x'
        assert data.test_set_y.name == 'test_set_y'
        assert np.asarray(data.dataset()[0][0].eval()) == 1
        assert np.asarray(data.dataset()[0][1].eval()) == 2
        assert np.asarray(data.dataset()[2][0].eval()) == 5
        assert np.asarray(data.dataset()[2][1].eval()) == 6
        data = Data(train_test_data, cast_y=True)
        assert np.asarray(data.train_set_x.eval()) == 1.
        assert np.asarray(data.train_set_y.eval()) == 2.
        assert data.valid_set_x is None
        assert data.valid_set_y is None
        assert np.asarray(data.test_set_x.eval()) == 5.
        assert np.asarray(data.test_set_y.eval()) == 6.
        assert data.train_set_y.eval().dtype == 'int32'
        assert data.test_set_y.eval().dtype == 'int32'
        assert data.train_set_x.name == 'train_set_x'
        assert data.test_set_x.name == 'test_set_x'
        assert np.asarray(data.dataset()[0][0].eval()) == 1
        assert np.asarray(data.dataset()[0][1].eval()) == 2
        assert np.asarray(data.dataset()[2][0].eval()) == 5
        assert np.asarray(data.dataset()[2][1].eval()) == 6
