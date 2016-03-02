# -*- coding: UTF-8 -*-
import time
import numpy as np
import theano


def test_to_float_X():
    from dl.utils import to_float_X
    x = np.asarray(np.random.normal(size=(10, 5)))
    assert to_float_X(x).dtype == theano.config.floatX


def test_shared_variable():
    from dl.utils import shared_variable
    x = np.asarray(np.random.normal(size=(10, 5)))
    assert isinstance(shared_variable(x), theano.compile.SharedVariable)
    assert shared_variable(None) is None


def test_format_sec():
    from dl.utils import format_sec
    s = 1*24*60*60 + 23*60*60 + 45*60 + 19 + 0.3456
    assert format_sec(s) == '1 d 23 h 45 m 19 s'
    s = 23*60*60 + 45*60 + 19 + 0.3456
    assert format_sec(s) == '23 h 45 m 19 s'
    s = 45*60 + 19 + 0.3456
    assert format_sec(s) == '45 m 19 s'
    s = 19 + 0.3456
    assert format_sec(s) == '19.346 s'


def test_timer(capsys):
    from dl.utils import timer

    @timer('test_function')
    def func():
        time.sleep(1)
    func()
    out, err = capsys.readouterr()
    out_split = out.split()
    assert out_split[0] == 'test_function'
    assert out_split[1] == 'took'
    assert float(out_split[2]) >= 1.
    assert out_split[3] == 's'

