# -*- coding: UTF-8 -*-
import theano.tensor as T

from .utils import EPSILON
"""
Objectives function are Theano function computing differences between
prediction and target
"""


def mean_squared_error(prediction, target):
    """
    Mean squared error

    .. math:: \sum_{i=1}^{n_batch}{\sqrt{\sum_{i=1}^{n_feature}{(prediction_{i,j} - target_{i,j})²}}}

    Parameters
    ----------
    prediction : Theano tensor
        The predicted values
    target : Theano tensor
        The target values

    Returns
    -------


    """
    return T.mean(T.square(prediction - target), axis=-1)


def root_mean_squared_error(prediction, target):
    return T.sqrt(T.mean(T.square(prediction - target), axis=-1))


def mean_absolute_error(prediction, target):
    return T.mean(T.abs_(prediction - target), axis=-1)


def hinge(prediction, target):
    return T.mean(T.maximum(1. - target * prediction, 0.), axis=-1)


def binary_crossentropy(prediction, target):
    clip_pred = T.clip(prediction, EPSILON, 1 - EPSILON)
    return T.mean(T.nnet.binary_crossentropy(clip_pred, target), axis=-1)


def categorical_crossentropy(prediction, target):
    prediction /= prediction.sum(axis=-1, keepdims=True)
    prediction = T.clip(prediction, EPSILON, 1 - EPSILON)
    return T.mean(T.nnet.categorical_crossentropy(prediction, target), axis=-1)