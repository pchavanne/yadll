# -*- coding: UTF-8 -*-

import pytest

import dl


def test_adagrad_updates():
    pytest.raises(NotImplementedError, dl.updates.adagrad_updates, 'cost', 'params')


def test_adadelta_updates():
    pytest.raises(NotImplementedError, dl.updates.adadelta_updates, 'cost', 'params')


def test_rmsprop_updates():
    pytest.raises(NotImplementedError, dl.updates.rmsprop_updates, 'cost', 'params')


def test_hessian_free_updates():
    pytest.raises(NotImplementedError, dl.updates.hessian_free_updates, 'cost', 'params')
