# -*- coding: UTF-8 -*-
import cPickle
import gzip

import theano.tensor as T

from .utils import *


def one_hot_encoding(arr, N=None):
    """
    One hot encoding of a vector of integer categorical variables in a range [0..N].

    You can provide the higher category N or max(arr) will be used.

    Parameters
    ----------
    arr : numpy array
        array of integer in a range [0, N]
    N : `int`, optional
        Higher category

    Returns
    -------
        one hot encoding [0, 1, 0, 0]

    Examples
    --------
    >>> a = np.asarray([1, 0, 3])
    >>> one_hot_encoding(a)
    array([[ 0.,  1.,  0.,  0.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> one_hot_encoding(a, 5)
    array([[ 0.,  1.,  0.,  0.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.]])
    """
    n = N
    if N is None:
        n = np.max(arr)
    oh = np.zeros((arr.shape[0], n + 1))
    oh[np.arange(arr.shape[0]), arr] = 1
    return oh


def one_hot_decoding(mat):
    """
    decoding of a one hot matrix

    Parameters
    ----------
    mat : numpy matrix
        one hot matrix

    Returns
    -------
        vector of decoded value

    Examples
    --------
    >>> a = np.asarray([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    >>> one_hot_decoding(a)
    array([1, 0, 3])
    """
    return np.argmax(mat, axis=1)


class Data(object):
    """
    Load gziped pickled data.

    data is made of train_set, valid_set, test_set
    and  set_x, set_y = set

    Parameters
    ----------
    data : `string`
        data file name (with path)
    shared : `bool`
        theano shared variable
    borrow : `bool`
        theano borrowable variable
    cast_y : `bool`
        cast y to `intX`

    Methods
    -------
    dataset :
        return the dataset

    Examples
    --------
    Load data

    >>> yadll.data.Data('data/mnist/mnist.pkl.gz')

    """
    def __init__(self, data, shared=True, borrow=True, cast_y=True):
        if isinstance(data, str):
            f = gzip.open(data, 'rb')
            train_set, valid_set, test_set = cPickle.load(f)
            f.close()
            train_set_x, train_set_y = train_set
            valid_set_x, valid_set_y = valid_set
            test_set_x, test_set_y = test_set

        elif isinstance(data, list):
            if len(data) == 3:
                train_set, valid_set, test_set = data
                valid_set_x, valid_set_y = valid_set

            elif len(data) == 2:
                train_set, test_set = data
                valid_set_x, valid_set_y = None, None

            else:
                raise TypeError

            train_set_x, train_set_y = train_set
            test_set_x, test_set_y = test_set

        else:
                raise TypeError

        if shared:
            self.train_set_x = shared_variable(train_set_x, name='train_set_x',
                                               borrow=borrow)
            self.train_set_y = shared_variable(train_set_y, name='train_set_y',
                                               borrow=borrow)
            self.valid_set_x = shared_variable(valid_set_x, name='valid_set_x',
                                               borrow=borrow)
            self.valid_set_y = shared_variable(valid_set_y, name='valid_set_y',
                                               borrow=borrow)
            self.test_set_x = shared_variable(test_set_x, name='test_set_x',
                                              borrow=borrow)
            self.test_set_y = shared_variable(test_set_y, name='test_set_y',
                                              borrow=borrow)

        if cast_y:
            self.train_set_y = T.cast(self.train_set_y, intX)
            if self.valid_set_y is not None:
                self.valid_set_y = T.cast(self.valid_set_y, intX)
            self.test_set_y = T.cast(self.test_set_y, intX)

    def dataset(self):
        return [(self.train_set_x, self.train_set_y),
                (self.valid_set_x, self.valid_set_y),
                (self.test_set_x, self.test_set_y)]
