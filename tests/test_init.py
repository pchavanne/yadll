# -*- coding: UTF-8 -*-
import numpy as np

import dl

eps = 1e-2
shape = (1000, 1000)


def test_init():

    init_obj = dl.init.constant
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.all([w.shape.eval(), np.asarray(shape)])
    assert np.mean(w.get_value()) == 0.0

    init_obj = (dl.init.constant, {'value': 3})
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.all([w.shape.eval(), np.asarray(shape)])
    assert np.mean(w.get_value()) == 3.0

    init_obj = dl.init.uniform
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.max(w.get_value()) - 0.5) < eps
    assert np.abs(np.min(w.get_value()) - (-0.5)) < eps

    init_obj = (dl.init.uniform, {'scale': (-2.0, 2.0)})
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.max(w.get_value()) - 2.0) < eps
    assert np.abs(np.min(w.get_value()) - (-2.0)) < eps

    init_obj = dl.init.normal
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.std(w.get_value()) - 0.5) < eps

    # Glorot
    init_obj = (dl.init.glorot_uniform, {'gain': dl.activation.tanh})
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = 1 * np.sqrt(6. / (shape[0] + shape[1]))
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.max(w.get_value()) - scale) < eps
    assert np.abs(np.min(w.get_value()) - (-scale)) < eps

    init_obj = (dl.init.glorot_uniform, {'gain': dl.activation.sigmoid})
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = 4 * np.sqrt(6. / (shape[0] + shape[1]))
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.max(w.get_value()) - scale) < eps
    assert np.abs(np.min(w.get_value()) - (-scale)) < eps

    init_obj = (dl.init.glorot_normal, {'gain': dl.activation.tanh})
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = 1 * np.sqrt(2. / (shape[0] + shape[1]))
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.std(w.get_value()) - scale) < eps

    init_obj = (dl.init.glorot_normal, {'gain': dl.activation.sigmoid})
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = 4 * np.sqrt(2. / (shape[0] + shape[1]))
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.std(w.get_value()) - scale) < eps

    # He
    init_obj = dl.init.He_uniform
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = np.sqrt(6. / shape[0])
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.max(w.get_value()) - scale) < eps
    assert np.abs(np.min(w.get_value()) - (-scale)) < eps

    init_obj = dl.init.He_normal
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    scale = np.sqrt(2. / shape[0])
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.abs(np.std(w.get_value()) - scale) < eps

    init_obj = dl.init.orthogonal
    w = dl.init.initializer(init_obj=init_obj, shape=shape, name='w')
    assert np.abs(np.mean(w.get_value()) - 0.0) < eps
    assert np.allclose(np.dot(w.get_value(), w.get_value().T), np.eye(min(shape)), atol=1e-5)



