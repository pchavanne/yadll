# -*- coding: UTF-8 -*-
import pytest


class TestModel:
    @pytest.fixture
    def model(self):
        from dl.model import Model
        return Model

