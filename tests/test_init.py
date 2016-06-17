# -*- coding: UTF-8 -*-
import numpy as np

import yadll

eps = 1e-2
shape = (1000, 1000)


def test_init():

    init_obj = yadll.init.constant
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.all([w.shape.eval(), np.asarray(shape)])
    assert np.mean(w.get_value()) == 0.0

    init_obj = (yadll.init.constant, {'value': 3})
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.all([w.shape.eval(), np.asarray(shape)])
    assert np.mean(w.get_value()) == 3.0

    init_obj = yadll.init.uniform
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.max(w.get_value()) - 0.5) < eps
    assert np.abs(np.min(w.get_value()) - (-0.5)) < eps

    init_obj = (yadll.init.uniform, {'scale': (-2.0, 2.0)})
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.max(w.get_value()) - 2.0) < eps
    assert np.abs(np.min(w.get_value()) - (-2.0)) < eps

    init_obj = yadll.init.normal
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.std(w.get_value()) - 0.5) < eps

    # Glorot
    init_obj = (yadll.init.glorot_uniform, {'gain': yadll.activation.tanh})
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = 1 * np.sqrt(6. / (shape[0] + shape[1]))
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.max(w.get_value()) - scale) < eps
    assert np.abs(np.min(w.get_value()) - (-scale)) < eps

    init_obj = (yadll.init.glorot_uniform, {'gain': yadll.activation.sigmoid})
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = 4 * np.sqrt(6. / (shape[0] + shape[1]))
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.max(w.get_value()) - scale) < eps
    assert np.abs(np.min(w.get_value()) - (-scale)) < eps

    init_obj = (yadll.init.glorot_uniform, {'gain': yadll.activation.sigmoid})
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, fan=(500, 500), name='w')
    scale = 4 * np.sqrt(6. / (500 + 500))
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.max(w.get_value()) - scale) < eps
    assert np.abs(np.min(w.get_value()) - (-scale)) < eps

    init_obj = (yadll.init.glorot_normal, {'gain': yadll.activation.tanh})
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = 1 * np.sqrt(2. / (shape[0] + shape[1]))
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.std(w.get_value()) - scale) < eps

    init_obj = (yadll.init.glorot_normal, {'gain': yadll.activation.sigmoid})
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = 4 * np.sqrt(2. / (shape[0] + shape[1]))
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.std(w.get_value()) - scale) < eps

    init_obj = (yadll.init.glorot_normal, {'gain': yadll.activation.sigmoid})
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, fan=(500, 500), name='w')
    scale = 4 * np.sqrt(2. / (500 + 500))
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.std(w.get_value()) - scale) < eps

    # He
    init_obj = yadll.init.He_uniform
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = np.sqrt(6. / shape[0])
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.max(w.get_value()) - scale) < eps
    assert np.abs(np.min(w.get_value()) - (-scale)) < eps

    init_obj = yadll.init.He_normal
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = np.sqrt(2. / shape[0])
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.std(w.get_value()) - scale) < eps

    # Orthogonal
    init_obj = yadll.init.orthogonal
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.allclose(np.dot(w.get_value(), w.get_value().T), np.eye(min(shape)), atol=1e-5)

    init_obj = (yadll.init.orthogonal, {'gain': yadll.activation.relu})
    w = yadll.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.allclose(np.dot(w.get_value(), w.get_value().T), np.eye(min(shape)) * 2, atol=1e-5)


