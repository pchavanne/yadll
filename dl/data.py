# -*- coding: UTF-8 -*-

import theano.tensor as T
import cPickle
import gzip
from .utils import *


class Data(object):
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

        self.train_set_x = shared_variable(train_set_x, name='train_set_x', borrow=borrow)
        self.train_set_y = shared_variable(train_set_y, name='train_set_x', borrow=borrow)
        self.valid_set_x = shared_variable(valid_set_x, name='train_set_x', borrow=borrow)
        self.valid_set_y = shared_variable(valid_set_y, name='train_set_x', borrow=borrow)
        self.test_set_x = shared_variable(test_set_x, name='train_set_x', borrow=borrow)
        self.test_set_y = shared_variable(test_set_y, name='train_set_x', borrow=borrow)

        if cast_y:
            self.train_set_y = T.cast(self.train_set_y, intX)
            self.valid_set_y = T.cast(self.valid_set_y, intX)
            self.test_set_y = T.cast(self.test_set_y, intX)

    def dataset(self):
        return [(self.train_set_x, self.train_set_y),
                (self.valid_set_x, self.valid_set_y),
                (self.test_set_x, self.test_set_y)]
