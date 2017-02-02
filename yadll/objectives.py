# -*- coding: UTF-8 -*-
import theano.tensor as T

from .utils import EPSILON
"""
Objectives functions are Theano functions for building loss expressions that
quantify how far prediction and target distributions are.
"""


def mean_squared_error(prediction, target):
    r"""
    Mean Squared Error:

    .. math:: MSE_i = \frac{1}{n} \sum_{j}{(prediction_{i,j} - target_{i,j})^2}

    """
    return T.mean(T.square(prediction - target), axis=-1)


def root_mean_squared_error(prediction, target):
    r"""
    Root Mean Squared Error:

    .. math:: RMSE_i = \sqrt{\frac{1}{n} \sum_{j}{(target_{i,j} - prediction_{i,j})^2}}

    """
    return T.sqrt(T.mean(T.square(prediction - target), axis=-1))


def mean_absolute_error(prediction, target):
    r"""
    Mean Absolute Error:

    .. math:: MAE_i = \frac{1}{n} \sum_{j}{|target_{i,j} - prediction_{i,j}}

    """
    return T.mean(T.abs_(prediction - target), axis=-1)


def binary_hinge_error(prediction, target):
    r"""
    Binary Hinge Error:

    .. math:: BHE_i = \frac{1}{n} \sum_{j}{\max(1 - target_{i,j} * prediction_{i,j}, 0)}

    """
    return T.mean(T.maximum(1. - target * prediction, 0.), axis=-1)


def categorical_hinge_error(prediction, target):
    r"""
    Categorical Hinge Error:

    .. math:: CHE_i = \frac{1}{n} \sum_{j}{\max(1 - target_{i,j} * prediction_{i,j}, 0)}

    """
    return T.mean(T.maximum(1. - target * prediction, 0.), axis=-1)


def binary_crossentropy_error(prediction, target):
    r"""
    Binary Cross-entropy Error:

    .. math:: BCE_i = - \frac{1}{n} \sum_{j}{target_{i,j} * \log(prediction_{i,j}
        - (1 - target_{i,j}) * \log(1 - prediction_{i,j}))}

    """
    clip_pred = T.clip(prediction, EPSILON, 1 - EPSILON)
    return T.mean(T.nnet.binary_crossentropy(clip_pred, target), axis=-1)


def categorical_crossentropy_error(prediction, target):
    r"""
    Categorical Cross-entropy Error:

    .. math:: CCE_i = - \frac{1}{n} \sum_{j}{target_{i,j} * \log(prediction_{i,j})}

    """
    prediction /= prediction.sum(axis=-1, keepdims=True)
    prediction = T.clip(prediction, EPSILON, 1 - EPSILON)
    return T.mean(T.nnet.categorical_crossentropy(prediction, target), axis=-1)


def kullback_leibler_divergence(prediction, target):
    r"""
    Kullback Leibler Divergence:

    .. math:: KLD_i = \sum_{j}{target_{i,j}*\log{frac{target_{i,j}}{prediction_{i,j}}}

    """
    prediction = T.clip(prediction, EPSILON, 1)
    target = T.clip(target, EPSILON, 1)
    return T.sum(target * T.log(target/prediction), axis=-1)

# Aliases
mse = MSE = mean_squared_error
rmse = RMSE = root_mean_squared_error
mae = MAE = mean_absolute_error
bhe = BHE = binary_hinge_error
che = CHE = categorical_hinge_error
bce = BCE = binary_crossentropy_error
cce = CCE = categorical_crossentropy_error
kld = KLD = kullback_leibler_divergence


# metrics
def binary_accuracy(prediction, target):
    return T.mean(T.eq(prediction, T.round(target)))


def categorical_accuracy(prediction, target):
    return T.mean(T.eq(T.argmax(prediction, axis=-1), T.argmax(target, axis=-1)))


def binary_error(prediction, target):
    return T.mean(T.eq(prediction, T.round(target)))


def categorical_error(prediction, target):
    return T.mean(T.neq(T.argmax(prediction, axis=-1), T.argmax(target, axis=-1)))