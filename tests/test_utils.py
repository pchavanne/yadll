# -*- coding: UTF-8 -*-
import time
import numpy as np
import theano

import logging


def test_to_float_X():
    from yadll.utils import to_float_X
    x = np.asarray(np.random.normal(size=(10, 5)))
    assert to_float_X(x).dtype == theano.config.floatX


def test_shared_variable():
    from yadll.utils import shared_variable
    x = np.asarray(np.random.normal(size=(10, 5)))
    assert isinstance(shared_variable(x), theano.compile.SharedVariable)
    assert shared_variable(None) is None


def test_format_sec():
    from yadll.utils import format_sec
    s = 1*24*60*60 + 23*60*60 + 45*60 + 19 + 0.3456
    assert format_sec(s) == '1 d 23 h 45 m 19 s'
    s = 23*60*60 + 45*60 + 19 + 0.3456
    assert format_sec(s) == '23 h 45 m 19 s'
    s = 45*60 + 19 + 0.3456
    assert format_sec(s) == '45 m 19 s'
    s = 19 + 0.3456
    assert format_sec(s) == '19.346 s'


def test_timer(caplog):
    from yadll.utils import timer
    caplog.setLevel(logging.INFO)

    @timer('test_function')
    def func():
        time.sleep(1)

    func()
    last_record = list(caplog.records())[-1]
    msg = last_record.getMessage()
    msg_split = msg.split()
    assert msg_split[0] == 'test_function'
    assert msg_split[1] == 'took'
    assert float(msg_split[2]) >= 1.
    assert msg_split[3] == 's'

