# -*- coding: UTF-8 -*-
import cPickle
import gzip
import pytest
import numpy as np


class Testdata:
    @pytest.fixture
    def train_valid_test_data(self):
        return [[1., 2.], [3., 4.], [5., 6.]]

    @pytest.fixture
    def train_test_data(self):
        return [[1., 2.], [5., 6.]]

    @pytest.fixture
    def train_valid_test_data_file(self, train_valid_test_data):
        data_file = 'data.pkl'
        f = gzip.open(data_file, 'wb')
        cPickle.dump(train_valid_test_data, f)
        f.close()
        return data_file

    def test_train_valid_test_data_file(self, train_valid_test_data_file):
        from dl.data import Data
        data = Data(train_valid_test_data_file, cast_y=False)
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
        #assert (np.asarray(data.dataset()) == [[1., 2.], [3., 4.], [5., 6.]]).all()
        data = Data(train_valid_test_data_file, cast_y=True)
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

    def test_train_valid_test_data(self, train_valid_test_data):
        from dl.data import Data
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

    def test_train_test_data(self, train_test_data):
        from dl.data import Data
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

    def test_len_raise_error(self):
        from dl.data import Data
        with pytest.raises(TypeError):
            data = Data([[1, 2]], cast_y=False)

    def test_raise_error(self):
        from dl.data import Data
        with pytest.raises(TypeError):
            data = Data(1, cast_y=False)

