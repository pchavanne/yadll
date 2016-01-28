# -*- coding: UTF-8 -*-

import dl


def test_hyperparameters():
    hp = dl.hyperparameters.Hyperparameters()
    hp('param1', 1, [1, 2, 3])
    hp('param2', 20, [10, 20, 30])
    hp('param3', 300, [100, 200, 300])

    assert hp.param1 == 1
    assert hp.param2 == 20
    assert hp.param3 == 300

    hp_iterations = [[h.param1, h.param2, h.param3] for h in hp]
    assert hp_iterations == [[1, 10, 100], [2, 10, 100], [3, 10, 100],
                             [1, 20, 100], [2, 20, 100], [3, 20, 100],
                             [1, 30, 100], [2, 30, 100], [3, 30, 100],
                             [1, 10, 200], [2, 10, 200], [3, 10, 200],
                             [1, 20, 200], [2, 20, 200], [3, 20, 200],
                             [1, 30, 200], [2, 30, 200], [3, 30, 200],
                             [1, 10, 300], [2, 10, 300], [3, 10, 300],
                             [1, 20, 300], [2, 20, 300], [3, 20, 300],
                             [1, 30, 300], [2, 30, 300], [3, 30, 300]]

    assert hp.param1 == 3
    assert hp.param2 == 30
    assert hp.param3 == 300

    hp.reset()

    assert hp.param1 == 1
    assert hp.param2 == 20
    assert hp.param3 == 300

