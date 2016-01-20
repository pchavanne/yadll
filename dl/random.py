# -*- coding: UTF-8 -*-

import numpy as np

_rng = np.random


def get_rng():
    return _rng


def set_rng(new_rng):
    global _rng
    _rng = new_rng


