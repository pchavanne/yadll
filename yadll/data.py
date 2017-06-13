# -*- coding: UTF-8 -*-
import os
import cPickle
import gzip

import theano.tensor as T

from .utils import *


def normalize(x):
    x_min = x.min()
    x_max = x.max()
    z = apply_normalize(x, x_min, x_max)
    return z, x_min, x_max


def apply_normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def revert_normalize(z, x_min, x_max):
    return z * (x_max - x_min) + x_min


def standardize(x):
    x_mean = x.mean()
    x_std = x.std()
    z = apply_standardize(x, x_mean, x_std)
    return z, x_mean, x_std


def apply_standardize(x, x_mean, x_std):
    return (x - x_mean) / x_std


def revert_standardize(z, x_mean, x_std):
    return (z * x_std) + x_mean


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


def mnist_loader():
    datafile = 'mnist.pkl.gz'
    if not os.path.isfile(datafile):
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, datafile)
    f = gzip.open(datafile, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set
    return [(train_set_x, one_hot_encoding(train_set_y)),
            (valid_set_x, one_hot_encoding(valid_set_y)),
            (test_set_x, one_hot_encoding(test_set_y))]


def alphabet_loader(sequence):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    number_of_chars = len(alphabet)
    sequence_length = sequence  # 2
    sentences = [alphabet[i: i + sequence_length] for i in range(len(alphabet) - sequence_length)]
    next_chars = [alphabet[i + sequence_length] for i in range(len(alphabet) - sequence_length)]

    # Transform sequences and labels into 'one-hot' encoding
    X = np.zeros((len(sentences), sequence_length, number_of_chars), dtype=np.bool)
    y = np.zeros((len(sentences), number_of_chars), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, ord(char) - ord('a')] = 1
        y[i, ord(next_chars[i]) - ord('a')] = 1
    return [(X, y), (X, y), (X, y)]


class Data(object):
    """
    Data container.

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
        return the dataset as Theano shared variables
        [(train_set_x, train_set_y),
         (valid_set_x, valid_set_y),
         (test_set_x, test_set_y)]

    Examples
    --------
    Load data

    >>> yadll.data.Data('data/mnist/mnist.pkl.gz')

    """
    def __init__(self, data, preprocessing=None,
                 shared=True, borrow=True, cast_y=False):
        self.data = data
        #TODO: Check data input
        if len(data) == 3:
            train_set, valid_set, test_set = data
            train_set_x, train_set_y = train_set
            valid_set_x, valid_set_y = valid_set
            test_set_x, test_set_y = test_set
        if len(data) == 2:
            train_set, test_set = data
            train_set_x, train_set_y = train_set
            valid_set_x, valid_set_y = None, None
            test_set_x, test_set_y = test_set

        self.preprocessing = preprocessing
        if preprocessing == 'Normalize':
            train_set_x, self.min, self.max = normalize(train_set_x)
            test_set_x = apply_normalize(test_set_x, self.min, self.max)
            if valid_set_x:
                valid_set_x = apply_normalize(valid_set_x, self.min, self.max)

        if preprocessing == 'Standardize':
            train_set_x, self.mean, self.std = standardize(train_set_x)
            test_set_x = apply_standardize(test_set_x, self.mean, self.std)
            if valid_set_x:
                valid_set_x = apply_standardize(valid_set_x, self.mean, self.std)

        if shared:
            self.train_set_x = shared_variable(train_set_x, name='train_set_x', borrow=borrow)
            self.train_set_y = shared_variable(train_set_y, name='train_set_y', borrow=borrow)
            self.valid_set_x = shared_variable(valid_set_x, name='valid_set_x', borrow=borrow)
            self.valid_set_y = shared_variable(valid_set_y, name='valid_set_y', borrow=borrow)
            self.test_set_x = shared_variable(test_set_x, name='test_set_x', borrow=borrow)
            self.test_set_y = shared_variable(test_set_y, name='test_set_y', borrow=borrow)

        if cast_y:
            self.train_set_y = T.cast(self.train_set_y, intX)
            if self.valid_set_y is not None:
                self.valid_set_y = T.cast(self.valid_set_y, intX)
            self.test_set_y = T.cast(self.test_set_y, intX)

    def dataset(self):
        return [(self.train_set_x, self.train_set_y),
                (self.valid_set_x, self.valid_set_y),
                (self.test_set_x, self.test_set_y)]

    def shape(self):
        return [(data[0].shape, data[1].shape) for data in self.data]
