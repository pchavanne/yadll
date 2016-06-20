# -*- coding: UTF-8 -*-
import itertools


class Hyperparameters(object):
    """
    Container class for the hyperparameters.
    Define each parameters with a name and a default value and optionally
    a list of values that will be iterated over during a grid search.

    It create an iterable of all the different parameters values combination.

    Parameters
    ----------
    name : `string`, {'batch_size', 'n_epochs', 'learning_rate', 'l1_reg', 'l2_reg', 'patience'}
        The name of the hyperparameter.
    value : `float`
        The default value of the hyperparameter.
    range : `list` of `float`
        A list of values iterated over during the gris search

    Methods
    -------
    reset
        reset all hyperparameters to default values.

    Examples
    --------
    Define the hyperparameters of the model:

    >>> hp = Hyperparameters()   # Create an Hyperparameters instance
    >>> hp('batch_size', 500)    # Define an hyperparameter with its default value
    >>> hp('n_epochs', 1000, [10, 100, 1000, 1000])    # and range for the grid search

    Grid search on the hyperparameters space:

    >>> for param in hp:
    >>>     # Do something with this set of hyperparameters

    """
    def __init__(self):
        self.hp_value = dict()
        self.hp_default = dict()
        self.hp_range = dict()
        self.iteration = 0

    def __call__(self, name, value, range=None):
        self.__setattr__(name, value)
        self.hp_value[name] = value
        self.hp_default[name] = value
        self.hp_range[name] = range
        if not range:
            self.hp_range[name] = [value]
        product = [x for x in apply(itertools.product, self.hp_range.values())]
        self.hp_product = [dict(zip(self.hp_value.keys(), p)) for p in product]

    def __str__(self):
        return str(self.hp_value)

    def __iter__(self):
        return self

    def next(self):
        if self.iteration > len(self.hp_product) - 1:
            raise StopIteration
        self.hp_value = self.hp_product[self.iteration]
        for name, value in self.hp_value.iteritems():
            self.__setattr__(name, value)
        self.iteration += 1
        return self

    def reset(self):
        for name, value in self.hp_default.iteritems():
            self.__setattr__(name, value)
        self.iteration = 0
