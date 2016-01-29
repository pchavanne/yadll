# -*- coding: UTF-8 -*-

import pytest

import dl


def test_sgd_updates():
    # TODO implement sgd_updates test
    pass


def test_momentum_updates():
    # TODO implement momentum_updates test
    pass


def test_nesterov_momentum_updates():
    # TODO implement nesterov_momentum_updates test
    pass


def test_adagrad_updates():
    pytest.raises(NotImplementedError, dl.updates.adagrad_updates, 'cost', 'params')


def test_adadelta_updates():
    pytest.raises(NotImplementedError, dl.updates.adadelta_updates, 'cost', 'params')


def test_rmsprop_updates():
    pytest.raises(NotImplementedError, dl.updates.rmsprop_updates, 'cost', 'params')


def test_hessian_free_updates():
    pytest.raises(NotImplementedError, dl.updates.hessian_free_updates, 'cost', 'params')
