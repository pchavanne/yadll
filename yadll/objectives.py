# -*- coding: UTF-8 -*-
import theano.tensor as T

from .utils import EPSILON
"""
Objectives functions are Theano functions for building loss expressions that
quantify how far prediction and target distributions are.
"""


def mean_squared_error(prediction, target):
    r"""
    Mean Squared Error: MSE

    .. math:: MSE_i = \frac{1}{n} \sum_{j}{(prediction_{i,j} - target_{i,j})^2}

    """
    return T.mean(T.square(prediction - target), axis=-1)


def root_mean_squared_error(prediction, target):
    r"""
    Root Mean Squared Error: RMSE

    .. math:: RMSE_i = \sqrt{\frac{1}{n} \sum_{j}{(target_{i,j} - prediction_{i,j})^2}}

    """
    return T.sqrt(T.mean(T.square(prediction - target), axis=-1))


def mean_absolute_error(prediction, target):
    r"""
    Mean Absolute Error: MAE

    .. math:: MAE_i = \frac{1}{n} \sum_{j}{\big|{target_{i,j} - prediction_{i,j}\big|}

    """
    return T.mean(T.abs_(prediction - target), axis=-1)


def binary_hinge_error(prediction, target):
    r"""
    Binary Hinge Error: BHE
    .. math:: hinge_i = \frac{1}{n} \sum_{j}{\max(1. - target_{i,j} * prediction_{i,j}, 0.)}

    """
    return T.mean(T.maximum(1. - target * prediction, 0.), axis=-1)


def categorical_hinge_error(prediction, target):
    r"""
    Categorical Hinge Error: CHE
    .. math:: hinge_i = \frac{1}{n} \sum_{j}{\max(1. - target_{i,j} * prediction_{i,j}, 0.)}

    """
    return T.mean(T.maximum(1. - target * prediction, 0.), axis=-1)


def binary_crossentropy_error(prediction, target):
    r"""
    Binary Cross-entropy Error: BCE

    .. math:: BCE_i = \frac{1}{n} \sum_{j}{-(target_{i,j} * \log(prediction_{i,j})
        + (1 - target_{i,j}) * \log(1 - prediction_{i,j}))}

    """
    clip_pred = T.clip(prediction, EPSILON, 1 - EPSILON)
    return T.mean(T.nnet.binary_crossentropy(clip_pred, target), axis=-1)


def categorical_crossentropy_error(prediction, target):
    r"""
    Categorical Cross-entropy Error: CCE

    .. math:: MAE_i = \frac{1}{n} \sum_{j}{\big|{target_{i,j} - prediction_{i,j}\big|}

    """
    prediction /= prediction.sum(axis=-1, keepdims=True)
    prediction = T.clip(prediction, EPSILON, 1 - EPSILON)
    return T.mean(T.nnet.categorical_crossentropy(prediction, target), axis=-1)


def kullback_leibler_divergence(prediction, target):
    r"""
    Kullback Leibler Divergence: KLD

    .. math:: MAE_i = \frac{1}{n} \sum_{j}{\big|{target_{i,j} - prediction_{i,j}\big|}

    """
    prediction /= prediction.sum(axis=-1, keepdims=True)
    prediction = T.clip(prediction, EPSILON, 1 - EPSILON)
    return T.mean(T.nnet.categorical_crossentropy(prediction, target), axis=-1)

# Aliases
mse = MSE = mean_squared_error
rmse = RMSE = root_mean_squared_error
mae = MAE = mean_absolute_error
bhe = BHE = binary_hinge_error
che = CHE = categorical_hinge_error
bce = BCE = binary_crossentropy_error
cce = CCE = categorical_crossentropy_error
kld = KLD = kullback_leibler_divergence
