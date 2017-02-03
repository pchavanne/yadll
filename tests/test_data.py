# -*- coding: UTF-8 -*-
import cPickle
import gzip
import pytest
import numpy as np


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

    def test_len_raise_error(self):
        from yadll.data import Data
        with pytest.raises(TypeError):
            data = Data([[1, 2]], cast_y=False)

    def test_raise_error(self):
        from yadll.data import Data
        with pytest.raises(TypeError):
            data = Data(1, cast_y=False)

