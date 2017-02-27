# -*- coding: UTF-8 -*-
"""
All the neural network layers currently supported by yaddll.
"""
from .init import *
from .objectives import *
from .updates import *
from .utils import *


from theano.tensor.signal import pool
from theano.tensor.nnet import conv

import logging

logger = logging.getLogger(__name__)


class Layer(object):
    """
    Layer is the base class of any neural network layer.
    It has to be subclassed by any kind of layer.

    Parameters
    ----------
    incoming : a `Layer` , a `List` of `Layers` or a `tuple` of `int`
        The incoming layer, a list of incoming layers or the shape of the input layer
    name : `string`, optional
        The layer name. default name is the class name
        plus instantiation number i.e: 'DenseLayer 3'

    """
    n_instances = 0

    def __init__(self, incoming, name=None, **kwargs):
        """
        The base class that represent a single layer of any neural network.
        It has to be subclassed by any kind of layer.

        """
        self.id = kwargs.pop('id', self.get_id())
        if incoming is None:
            # incoming can be set to None to creat nested layers.
            self.input_shape = None
            self.input_layer = None
        elif isinstance(incoming, tuple):
            # incoming is a tuple for input layer
            self.input_shape = incoming
            self.input_layer = None
        elif isinstance(incoming, list):
            # incoming can be a list of layer
            self.input_shape = [inc if isinstance(inc, tuple) else inc.output_shape for inc in incoming]
            self.input_layer = [None if isinstance(inc, tuple) else inc for inc in incoming]
        else:
            # incoming is a layer
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        if name is None:
            self.name = self.__class__.__name__ + ' ' + str(self.id)
        self.params = []
        self.reguls = 0

    @classmethod
    def get_id(cls):
        cls.n_instances += 1
        return cls.n_instances

    def get_params(self):
        """
        Theano shared variables representing the parameters of this layer.

        Returns
        -------
            list of Theano shared variables that parametrize the layer
        """
        return self.params

    def get_reguls(self):
        """
        Theano expression representing the sum of the regulators of
        this layer.

        Returns
        -------
            Theano expression representing the sum of the regulators
         of this layer
        """
        return self.reguls

    @property
    def output_shape(self):
        """
        Compute the output shape of this layer given the input shape.

        Returns
        -------
            a tuple representing the shape of the output of this layer.

        Notes
        -----
        This method has to be overriden by new layer implementation or
        will return the input shape.
        """
        return self.input_shape

    def get_output(self, **kwargs):
        """
        Return the output of this layer

        Raises
        ------
        NotImplementedError
            This method has to be overriden by new layer implementation.
        """
        raise NotImplementedError

    def to_conf(self):
        # conf = self.__dict__.copy()
        # for key in ['params', 'reguls']:
        #     conf.pop(key, None)
        # if conf['input_layer']:
        #     conf['input_layer'] = conf['input_layer'].name
        # if 'activation' in conf:
        #     conf['activation'] = conf['activation'].__name__
        # if 'hyperparameters' in conf:
        #     conf['hp'] = conf.pop('hyperparameters').to_conf()
        # conf['type'] = self.__class__.__name__
        conf = {'type': self.__class__.__name__,
                'id': self.id,
                'name': self.name,
                'input_shape': self.input_shape}
        if self.input_layer is None:
            conf['input_layer'] = None
        elif isinstance(self.input_layer, list):
            conf['input_layer'] = [layer.name for layer in self.input_layer]
        else:
            conf['input_layer'] = self.input_layer.name
        return conf

    # def from_conf(self, conf):
    #     self.__dict__.update(conf)
    #     if 'hp' in conf:
    #         for k, v in conf['hp'].iteritems():
    #             self.hp(k, v)


class InputLayer(Layer):
    """
    Input layer of the data, it has no parameters, it just shapes the data as
    the input for any network. A ::class:`InputLayer` is always the first layer of any network.
    """
    n_instances = 0

    def __init__(self, input_shape, input=None, **kwargs):
        """
        The input layer of any network

        Parameters
        ----------
        shape : `tuple` of `int`
            The shape of the input layer the first element is the batch size
            and can be set to None.
        input : `Theano shared Variables`, optional
            The input data of the network, used to train the model
        """
        super(InputLayer, self).__init__(input_shape, **kwargs)
        self.input = input

    def get_output(self, **kwargs):
        return self.input

    def to_conf(self):
        conf = super(InputLayer, self).to_conf()
        return conf


class ReshapeLayer(Layer):
    """
    Reshape the incoming layer to the output_shape.
    """
    n_instances = 0

    def __init__(self, incoming, output_shape=None, **kwargs):
        super(ReshapeLayer, self).__init__(incoming, **kwargs)
        self.reshape_shape = output_shape

    @property
    def output_shape(self):
        return self.reshape_shape

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        if self.reshape_shape[0] is None:
            lst = list(self.reshape_shape)
            lst[0] = T.shape(X)[0]
            self.reshape_shape = tuple(lst)
        return X.reshape(self.reshape_shape)

    def to_conf(self):
        conf = super(ReshapeLayer, self).to_conf()
        conf['output_shape'] = self.output_shape
        return conf


class FlattenLayer(Layer):
    """
    Reshape layers back to flat
    """
    n_instances = 0

    def __init__(self, incoming, n_dim=2, **kwargs):
        super(FlattenLayer, self).__init__(incoming, **kwargs)
        self.n_dim = n_dim

    @property
    def output_shape(self):
        return self.input_shape[0], np.prod(self.input_shape[1:])

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return X.flatten(self.n_dim)

    def to_conf(self):
        conf = super(FlattenLayer, self).to_conf()
        conf['n_dim'] = self.n_dim
        return conf


class Activation(Layer):
    """
    Apply activation function to previous layer
    """
    n_instances = 0

    def __init__(self, incoming, activation=linear, **kwargs):
        super(Activation, self).__init__(incoming, **kwargs)
        self.activation = get_activation(activation)

    @property
    def output_shape(self):
        return self.input_shape[0], np.prod(self.input_shape[1:])

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return self.activation(X)

    def to_conf(self):
        conf = super(Activation, self).to_conf()
        conf['activation'] = activation_to_conf(self.activation)
        return conf

    def __getstate__(self):
        if hasattr(self.activation, '__call__'):
            dic = self.__dict__.copy()
            dic['activation'] = activation_to_conf(self.activation)
            return dic

    def __setstate__(self, dic):
        self.__dict__.update(dic)
        self.activation = get_activation(self.activation)


class DenseLayer(Layer):
    """
    Fully connected neural network layer
    """
    n_instances = 0

    def __init__(self, incoming, n_units, W=glorot_uniform, b=constant,
                 activation=tanh, l1=None, l2=None, **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.n_units = n_units
        self.shape = (self.input_shape[1], n_units)
        if isinstance(W, theano.compile.SharedVariable):
            self.W = W
        else:
            self.W = initializer(W, shape=self.shape, name='W')
        self.params.append(self.W)
        if isinstance(b, theano.compile.SharedVariable):
            self.b = b
        else:
            self.b = initializer(b, shape=(self.shape[1],), name='b')
        self.params.append(self.b)
        self.activation = get_activation(activation)
        self.l1 = l1
        self.l2 = l2
        if l1 and l1 != 0:
            self.reguls += l1 * T.mean(T.abs_(self.W))
        if l2 and l2 != 0:
            self.reguls += l2 * T.mean(T.sqr(self.W))

    @property
    def output_shape(self):
        return self.input_shape[0], self.shape[1]

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return self.activation(T.dot(X, self.W) + self.b)

    def to_conf(self):
        conf = super(DenseLayer, self).to_conf()
        conf['n_units'] = self.n_units
        conf['activation'] = activation_to_conf(self.activation)
        conf['l1'] = self.l1
        conf['l2'] = self.l2
        return conf

    def __getstate__(self):
        if hasattr(self.activation, '__call__'):
            dic = self.__dict__.copy()
            dic['activation'] = activation_to_conf(self.activation)
            return dic

    def __setstate__(self, dic):
        self.__dict__.update(dic)
        self.activation = get_activation(self.activation)


class UnsupervisedLayer(DenseLayer):
    """
    Base class for all unsupervised layers.
    Unsupervised layers are pre-trained against its own input.

    """
    n_instances = 0

    def __init__(self, incoming, n_units, hyperparameters, **kwargs):
        super(UnsupervisedLayer, self).__init__(incoming, n_units, **kwargs)
        self.hp = hyperparameters
        self.unsupervised_params = list(self.params)

    def get_encoded_input(self, **kwargs):
        raise NotImplementedError

    def get_unsupervised_cost(self, **kwargs):
        raise NotImplementedError

    @timer(' Pre-training the layer')
    def unsupervised_training(self, x, train_set_x):
        logger.info('... Pre-training the layer: %s' % self.name)
        index = T.iscalar('index')
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / self.hp.batch_size
        unsupervised_cost = self.get_unsupervised_cost()
        if isinstance(unsupervised_cost, tuple):
            cost = unsupervised_cost[0]
            updates = unsupervised_cost[1]
        else:
            cost = unsupervised_cost
            updates = sgd(cost, self.unsupervised_params, self.hp.learning_rate)
        pretrain = theano.function(inputs=[index], outputs=cost, updates=updates,
                                   givens={x: train_set_x[index * self.hp.batch_size: (index + 1) * self.hp.batch_size]})
        for epoch in xrange(self.hp.n_epochs):
            c = []
            for minibatch_index in xrange(n_train_batches):
                c.append(pretrain(minibatch_index))
            logger.info('Layer: %s, pre-training epoch %d, cost %d' % (self.name, epoch, np.mean(c)))

    def to_conf(self):
            conf = super(UnsupervisedLayer, self).to_conf()
            conf['hyperparameters'] = self.hp.to_conf()
            return conf


class LogisticRegression(DenseLayer):
    """
    Dense layer with softmax activation

    References
    ----------
    .. [1] http://deeplearning.net/tutorial/logreg.html
    """
    n_instances = 0

    def __init__(self, incoming, n_class, W=constant, activation=softmax, **kwargs):
        super(LogisticRegression, self).__init__(incoming, n_class, W=W,
                                                 activation=activation, **kwargs)

    def to_conf(self):
        conf = super(LogisticRegression, self).to_conf()
        conf['n_class'] = conf.pop('n_units')
        return conf


class Dropout(Layer):
    """
    Dropout layer
    """
    n_instances = 0

    def __init__(self, incoming, corruption_level=0.5, **kwargs):
        super(Dropout, self).__init__(incoming, **kwargs)
        self.p = 1 - corruption_level

    def get_output(self, stochastic=True, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.input_shape[0] is None:
            lst = list(self.input_shape)
            lst[0] = T.shape(X)[0]
            self.input_shape = tuple(lst)
        if self.p != 1 and stochastic:
            X = X * T_rng.binomial(self.input_shape, n=1, p=self.p, dtype=floatX)
        return X

    def to_conf(self):
        conf = super(Dropout, self).to_conf()
        conf['corruption_level'] = 1 - self.p
        return conf


class Dropconnect(DenseLayer):
    """
    DropConnect layer
    """
    n_instances = 0

    def __init__(self, incoming, n_units, corruption_level=0.5, **kwargs):
        super(Dropconnect, self).__init__(incoming, n_units, **kwargs)
        self.p = 1 - corruption_level

    def get_output(self, stochastic=True, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.p != 1 and stochastic:
            self.W = self.W * T_rng.binomial(self.shape, n=1, p=self.p, dtype=floatX)
        return self.activation(T.dot(X, self.W) + self.b)

    def to_conf(self):
        conf = super(Dropconnect, self).to_conf()
        conf['corruption_level'] = 1 - self.p
        return conf


class PoolLayer(Layer):
    """
    Pooling layer, default is maxpooling
    """
    n_instances = 0

    def __init__(self, incoming, pool_size, stride=None, ignore_border=True,
                 pad=(0, 0), mode='max', **kwargs):
        super(PoolLayer, self).__init__(incoming, **kwargs)
        self.pool_size = pool_size
        self.stride = stride    # If st is None, it is considered equal to ds
        self.ignore_border = ignore_border
        self.pad = pad
        self.mode = mode    # {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}

    def pool(self, input, ws):
        return pool.pool_2d(input=input, ws=ws, st=self.stride, ignore_border=self.ignore_border,
                            pad=self.pad, mode=self.mode)

    @property
    def output_shape(self):
        return (self.input_shape[0],
                self.input_shape[1],
                self.input_shape[2] / self.pool_size[0],
                self.input_shape[3] / self.pool_size[1])

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        return self.pool(input=X, ws=self.pool_size)

    def to_conf(self):
        conf = super(PoolLayer, self).to_conf()
        conf['pool_size'] = self.pool_size
        conf['pool_size'] = self.stride
        conf['ignore_border'] = self.ignore_border
        conf['pad'] = self.pad
        conf['mode'] = self.mode
        return conf


class ConvLayer(Layer):
    """
    Convolutional layer
    """
    n_instances = 0

    def __init__(self, incoming, image_shape=None, filter_shape=None, W=glorot_uniform,
                 border_mode='valid', subsample=(1, 1), l1=None, l2=None, pool_scale=None, **kwargs):
        super(ConvLayer, self).__init__(incoming, **kwargs)
        assert image_shape[1] == filter_shape[1]
        self.image_shape = image_shape      # (batch size, num input feature maps, image height, image width)
        self.filter_shape = filter_shape    # (number of filters, num input feature maps, filter height, filter width)
        self.border_mode = border_mode      # {'valid', 'full'}
        self.subsample = subsample
        self.fan_in = np.prod(filter_shape[1:])
        self.fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        self.pool_scale = pool_scale
        if pool_scale:
            self.fan_out /= np.prod(pool_scale)
        self.W = initializer(W, shape=self.filter_shape, fan=(self.fan_in, self.fan_out), name='W')
        self.params.append(self.W)
        self.l1 = l1
        self.l2 = l2
        if l1:
            self.reguls += l1 * T.mean(T.abs_(self.W))
        if l2:
            self.reguls += l2 * T.mean(T.sqr(self.W))

    @property
    def output_shape(self):
        return (self.input_shape[0],
                self.filter_shape[0],
                self.image_shape[2] - self.filter_shape[2] + 1,
                self.image_shape[3] - self.filter_shape[3] + 1)

    def conv(self, input, filters, image_shape, filter_shape):
        return conv.conv2d(input=input, filters=filters, image_shape=image_shape,
                           filter_shape=filter_shape, border_mode=self.border_mode, subsample=self.subsample)

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        return self.conv(input=X, filters=self.W, image_shape=self.image_shape,
                         filter_shape=self.filter_shape)

    def to_conf(self):
        conf = super(ConvLayer, self).to_conf()
        conf['image_shape'] = self.image_shape
        conf['filter_shape'] = self.filter_shape
        conf['border_mode'] = self.border_mode
        conf['subsample'] = self.subsample
        conf['l1'] = self.l1
        conf['l2'] = self.l2
        conf['pool_scale'] = self.pool_scale
        return conf


class ConvPoolLayer(ConvLayer, PoolLayer):
    """
    Convolutional and pooling layer

    References
    ----------
    .. [1] http://deeplearning.net/tutorial/lenet.html
    """
    n_instances = 0

    def __init__(self, incoming, pool_size, image_shape=None, filter_shape=None,
                 b=constant, activation=tanh, **kwargs):
        super(ConvPoolLayer, self).__init__(incoming, pool_size=pool_size, image_shape=image_shape,
                                            filter_shape=filter_shape, pool_scale=pool_size, **kwargs)
        self.b = initializer(b, shape=(self.filter_shape[0],), name='b')
        self.params.append(self.b)
        self.activation = get_activation(activation)

    @property
    def output_shape(self):
        return (self.input_shape[0],
                self.filter_shape[0],
                (self.image_shape[2] - self.filter_shape[2] + 1) / self.pool_size[0],
                (self.image_shape[3] - self.filter_shape[3] + 1) / self.pool_size[1])

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        conv_X = self.conv(input=X, filters=self.W, image_shape=self.image_shape,
                           filter_shape=self.filter_shape)
        pool_X = self.pool(input=conv_X, ws=self.pool_size)
        return self.activation(pool_X + self.b.dimshuffle('x', 0, 'x', 'x'))

    def to_conf(self):
        conf = super(ConvLayer, self).to_conf()
        conf['activation'] = activation_to_conf(self.activation)
        return conf


class AutoEncoder(UnsupervisedLayer):
    """
    Autoencoder

    References
    ----------
    .. [1] http://deeplearning.net/tutorial/dA.html
    """
    n_instances = 0

    def __init__(self, incoming, n_units, hyperparameters, corruption_level=0.0,
                 W=(glorot_uniform, {'gain': sigmoid}), b_prime=constant,
                 sigma=None, contraction_level=None, **kwargs):
        super(AutoEncoder, self).__init__(incoming, n_units, hyperparameters, W=W, **kwargs)
        self.W_prime = self.W.T
        if isinstance(b_prime, theano.compile.SharedVariable):
            self.b_prime = b_prime
        else:
            self.b_prime = initializer(b_prime, shape=(self.shape[0],), name='b_prime')
        self.unsupervised_params.append(self.b_prime)
        self.p = 1 - corruption_level
        self.sigma = sigma  # standard deviation if gaussian noise.
        self.contraction_level = contraction_level  # for Contractive Autoencoders

    def get_encoded_input(self, **kwargs):
        X = self.input_layer.get_output(stochastic=False, **kwargs)
        if self.p > 0:
            if self.sigma:
                X = X + T_rng.normal((self.hp.batch_size, self.input_shape[1]), avg=0.0, std=self.sigma, dtype=floatX)
            else:
                X = X * T_rng.binomial((self.hp.batch_size, self.input_shape[1]), n=1, p=self.p, dtype=floatX)
        Y = sigmoid(T.dot(X, self.W) + self.b)
        Z = sigmoid(T.dot(Y, self.W_prime) + self.b_prime)
        return Y, Z

    def get_unsupervised_cost(self, **kwargs):
        X = self.input_layer.get_output(stochastic=False, **kwargs)
        Y, Z = self.get_encoded_input(**kwargs)
        cost = T.mean(categorical_crossentropy_error(Z, X))
        if self.contraction_level:
            # For sigmoid: J = Y * (1 - Y) * W
            # For tanh: J = (1 + Y) * (1 - Y) * W
            J = T.reshape(Y * (1 - Y), (self.hp.batch_size, 1, self.shape[1])) \
                * T.reshape(self.W, (1, self.shape[0], self.shape[1]))
            Lj = T.sum(J**2) / self.hp.batch_size
            cost = cost + self.contraction_level * T.mean(Lj)
        return cost

    def to_conf(self):
        conf = super(AutoEncoder, self).to_conf()
        conf['corruption_level'] = 1 - self.p
        conf['sigma'] = self.sigma
        conf['contraction_level'] = self.contraction_level
        return conf


class RBM(UnsupervisedLayer):
    """
    Restricted Boltzmann Machines

    References
    ----------
    .. [1] http://deeplearning.net/tutorial/rbm.html
    """
    n_instances = 0

    def __init__(self, incoming, n_units, hyperparameters, W=glorot_uniform,
                 b_hidden=constant, activation=sigmoid, **kwargs):
        super(RBM, self).__init__(incoming, n_units, hyperparameters, W=W,
                                  activation=activation, **kwargs)
        if isinstance(b_hidden, theano.compile.SharedVariable):
            self.b_hidden = b_hidden
        else:
            self.b_hidden = initializer(b_hidden, shape=(self.shape[0],), name='b_hidden')
        self.unsupervised_params.append(self.b_hidden)

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.b
        vbias_term = T.dot(v_sample, self.b_hidden)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def prop_up(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.b
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_min = self.prop_up(v0_sample)
        h1_sample = T_rng.binomial(size=h1_min.shape, n=1, p=h1_min, dtype=floatX)
        return [pre_sigmoid_h1, h1_min, h1_sample]

    def prop_down(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.b_hidden
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_min = self.prop_down(h0_sample)
        v1_sample = T_rng.binomial(size=v1_min.shape, n=1, p=v1_min, dtype=floatX)
        return [pre_sigmoid_v1, v1_min, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        cross_entropy = T.mean(T.sum(X * T.log(sigmoid(pre_sigmoid_nv)) +
                (1 - X) * T.log(1 - sigmoid(pre_sigmoid_nv)), axis=1))
        # cross_entropy = binary_crossentropy(X, sigmoid(pre_sigmoid_nv))
        # TODO compare the two cross entropies and check without updates
        return cross_entropy

    def get_pseudo_likelihood_cost(self, updates, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(X)
        fe_xi = self.free_energy(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)
        cost = T.mean(self.shape[0] * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.shape[0]
        return cost

    def get_encoded_input(self, stochastic=False, **kwargs):
        raise NotImplementedError

    def get_unsupervised_cost(self, persistent=None, k=1, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(X)
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        ([pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates) = \
            theano.scan(fn=self.gibbs_hvh,
                        outputs_info=[None, None, None, None, None, chain_start],
                        n_steps=k)
        chain_end = nv_samples[-1]
        cost = T.mean(self.free_energy(X)) - T.mean(self.free_energy(chain_end))
        gparams = T.grad(cost, self.unsupervised_params, consider_constant=[chain_end])
        for gparam, param in zip(gparams, self.unsupervised_params):
            updates[param] = param - gparam * T.cast(self.hp.learning_rate, dtype=floatX)
        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])
        return monitoring_cost, updates

    def to_conf(self):
        conf = super(RBM, self).to_conf()
        return conf


class BatchNormalization(Layer):
    r"""
    Normalize the input layer over each mini-batch according to [1]_:

    .. math::
        \hat{x} = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}}

        y = \gamma * \hat{x} + \beta

    .. warning::
       When a BatchNormalization layer is used the batch size has to be given at compile time.
       You can not use None as the first dimension anymore.
       Prediction has to be made on the same batch size.

    References
    ----------

    .. [1] http://jmlr.org/proceedings/papers/v37/ioffe15.pdf
    """
    n_instances = 0

    def __init__(self, incoming, axis=-2, alpha=0.1, epsilon=1e-5, beta=True, **kwargs):
        super(BatchNormalization, self).__init__(incoming, **kwargs)
        self.axis = axis
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = self.gamma = self.mean = self.var = None
        if self.input_shape is not None:
            self.init_params(self.input_shape, beta=beta)

    def init_params(self, input_shape, beta):
        self.gamma = initializer(constant, shape=input_shape, value=1, name='gamma')
        self.params.append(self.gamma)
        self.beta = initializer(constant, shape=(input_shape[1],), value=0, name='beta')
        if beta:
            self.params.append(self.beta)
        self.mean = initializer(constant, shape=input_shape, value=0, name='mean')
        self.var = initializer(constant, shape=input_shape, value=1, name='var')

    def get_output(self, stochastic=True, **kwargs):
        x = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if stochastic:
            mean = T.mean(x, axis=self.axis)                           # mini-batch mean
            var = T.var(x, axis=self.axis)                             # mini-batch variance
            self.mean = self.alpha * self.mean + (1 - self.alpha) * mean
            self.var = self.alpha * self.var + (1 - self.alpha) * var
        else:
            mean = self.mean
            var = self.var
        x_hat = (x - mean) / T.sqrt(var + self.epsilon)                 # normalize
        y = self.gamma * x_hat + self.beta                              # scale and shift
        return y

    def to_conf(self):
        conf = super(BatchNormalization, self).to_conf()
        conf['axis'] = self.axis
        conf['alpha'] = self.alpha
        conf['epsilon'] = self.epsilon
        conf['beta'] = self.beta
        return conf


class RNN(Layer):
    r"""
    Recurrent Neural Network

    .. math ::
        h_t = \sigma(x_t.W + h_{t-1}.U + b)

    References
    ----------

    .. [1] http://deeplearning.net/tutorial/rnnslu.html
    .. [2] https://arxiv.org/pdf/1602.06662.pdf
    .. [3] https://arxiv.org/pdf/1511.06464.pdf
    """
    n_instances = 0

    def __init__(self, incoming, n_units, n_out=None, activation=sigmoid,
                 last_only=True, grad_clipping=0, go_backwards=False, allow_gc=False, **kwargs):
        super(RNN, self).__init__(incoming, **kwargs)
        self.allow_gc = allow_gc
        self.go_backwards = go_backwards
        self.grad_clipping = grad_clipping
        self.last_only = last_only
        self.activation = get_activation(activation)

        self.n_feature = self.input_shape[2]  # (n_batch, n_time_steps, n_dim)
        self.n_hidden = n_units
        if n_out is None:
            self.n_out = n_units
        else:
            self.n_out = n_out
        self.W = orthogonal(shape=(self.n_feature, self.n_out), name='W')
        self.U = orthogonal(shape=(self.n_hidden, self.n_out), name='U')
        self.b = uniform(shape=self.n_out, scale=(0, 1.), name='b')
        # trainable parameters
        self.params.extend([self.W, self.U, self.b])
        # Non sequence for the scan operator
        self.non_seq = [self.U]

    @property
    def output_shape(self):
        if self.last_only:
            out_shape = (self.input_shape[0], self.n_out)
        else:
            out_shape = (self.input_shape[0], self.input_shape[1], self.n_out)
        return out_shape

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)

        if X.ndim > 3:
            X = T.flatten(X, 3)
        # (n_batch, n_time_steps, n_dim) ->  (n_time_steps, n_batch, n_dim)
        X = X.dimshuffle(1, 0, 2)
        n_batch = X.shape[1]
        # Input dot product is outside of the scan
        X = T.dot(X, self.W) + self.b

        c0 = T.ones((n_batch, self.n_hidden), dtype=floatX)
        h0 = self.activation(c0)

        def one_step(x_t, h_tm1, *args):
            # pre-activation
            pre_act = x_t + T.dot(h_tm1, self.U)
            # Clip gradients
            if self.grad_clipping:
                pre_act = theano.gradient.grad_clip(pre_act, -self.grad_clipping, self.grad_clipping)
            h_t = self.activation(pre_act)

            return h_t

        h_vals, _ = theano.scan(fn=one_step,
                                sequences=X,
                                outputs_info=h0,
                                non_sequences=self.non_seq,
                                go_backwards=self.go_backwards,
                                allow_gc=self.allow_gc,
                                strict=True)
        if self.last_only:
            h_vals = h_vals[-1]
        else:
            h_vals = h_vals.dimshuffle(1, 0, 2)
            if self.go_backwards:
                h_vals = h_vals[:, ::-1]

        return h_vals

    def to_conf(self):
        conf = super(RNN, self).to_conf()
        return conf


class LSTM(Layer):
    r"""
    Long Short Term Memory

    .. math ::
        i_t &= \sigma(x_t.W_i + h_{t-1}.U_i + b_i)\\
        f_t &= \sigma(x_t.W_f + h_{t-1}.U_f + b_f)\\
        \tilde{C_t} &= \tanh(x_t.W_c + h_{t-1}.U_c + b_c)\\
        C_t &= f_t * C_{t-1} + i_t * \tilde{C_t}\\
        o_t &= \sigma(x_t.W_o + h_{t-1}.U_o + b_o)\\
        h_t &= o_t * \tanh(C_t)
        \text{with Peephole connections:}\\
        i_t &= \sigma(x_t.W_i + h_{t-1}.U_i + C_{t-1}.P_i + b_i)\\
        f_t &= \sigma(x_t.W_f + h_{t-1}.U_f + C_{t-1}.P_f + b_f)\\
        \tilde{C_t} &= \tanh(x_t.W_c + h_{t-1}.U_c + b_c)\\
        C_t &= f_t * C_{t-1} + i_t * \tilde{C_t}\\
        o_t &= \sigma(x_t.W_o + h_{t-1}.U_o + C_t.P_o + b_o)\\
        h_t &= o_t * \tanh(C_t)\\
        \text{with tied forget and input gates:}\\
        C_t &= f_t * C_{t-1} + (1 - f_t) * \tilde{C_t}\\

    Parameters
    ----------
    incoming : a `Layer`
        The incoming layer with an output_shape = (n_batches, n_time_steps, n_dim)
    n_units : int
        n_hidden = n_input_gate = n_forget_gate = n_cell_gate = n_output_gate = n_units
        All gates have the same number of units
    n_out : int
        number of output units
    peephole : boolean default is False
        use peephole connections.
    tied_i : boolean default is false
        tie input and forget gate
    activation : `yadll.activations` function default is `yadll.activations.tanh`
        activation function
    last_only : boolean default is True
        set to true if you only need the last element of the output sequence.
        Theano will optimize graph.

    References
    ----------
    .. [1] http://deeplearning.net/tutorial/lstm.html
    .. [2] http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php
    .. [3] http://people.idsia.ch/~juergen/lstm/
    .. [4] http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    .. [5] https://arxiv.org/pdf/1308.0850v5.pdf
    """
    n_instances = 0

    def __init__(self, incoming, n_units, peepholes=False, tied_i_f=False, activation=tanh,
                 last_only=True, grad_clipping=0, go_backwards=False, allow_gc=False, **kwargs):
        super(LSTM, self).__init__(incoming, **kwargs)
        self.allow_gc = allow_gc
        self.grad_clipping = grad_clipping
        self.go_backwards = go_backwards
        self.last_only = last_only
        self.peepholes = peepholes    # input and forget gates layers look at the cell state
        self.tied = tied_i_f        # only input new values to the state when we forget something
        self.activation = get_activation(activation)
        self.n_feature = self.input_shape[2]  # (n_batch, n_time_steps, n_dim)
        self.n_units = self.n_hidden = self.n_ig = self.n_fg = self.n_cg = self.n_og = n_units
        # input gate
        self.W_i = orthogonal(shape=(self.n_feature, self.n_ig), name='W_i')
        self.U_i = orthogonal(shape=(self.n_hidden, self.n_ig), name='U_i')
        self.b_i = uniform(shape=(self.n_ig,), scale=(-0.5, .5), name='b_i')
        # forget gate
        self.W_f = orthogonal(shape=(self.n_feature, self.n_fg), name='W_f')
        self.U_f = orthogonal(shape=(self.n_hidden, self.n_fg), name='U_f')
        self.b_f = uniform(shape=(self.n_fg,), scale=(0, 1.), name='b_f')
        # cell gate
        self.W_c = orthogonal(shape=(self.n_feature, self.n_cg), name='W_c')
        self.U_c = orthogonal(shape=(self.n_hidden, self.n_cg), name='U_c')
        self.b_c = constant(shape=(self.n_cg,), name='b_c')
        # output gate
        self.W_o = orthogonal(shape=(self.n_feature, self.n_og), name='W_o')
        self.U_o = orthogonal(shape=(self.n_hidden, self.n_og), name='U_o')
        self.b_o = uniform(shape=(self.n_ig,), scale=(-0.5, .5), name='b_o')
        # Row representation
        self.W = T.concatenate([self.W_i, self.W_f, self.W_c, self.W_o], axis=1)
        self.U = T.concatenate([self.U_i, self.U_f, self.U_c, self.U_o], axis=1)
        self.b = T.concatenate([self.b_i, self.b_f, self.b_c, self.b_o], axis=0)
        # Non sequence for the scan operator
        self.non_seq = [self.U]

        if True:
            self.params.extend([self.W_i, self.U_i, self.b_i,
                                self.W_f, self.U_f, self.b_f,
                                self.W_c, self.U_c, self.b_c,
                                self.W_o, self.U_o, self.b_o])
        else:
            self.params.extend([self.W, self.U, self.b])

        if peepholes:
            self.P_i = orthogonal(shape=(self.n_cg, self.n_ig), name='P_i')
            self.P_f = orthogonal(shape=(self.n_cg, self.n_fg), name='P_f')
            self.P_c = constant((self.n_cg, self.n_cg), name='P_c')
            self.P_o = orthogonal(shape=(self.n_cg, self.n_og), name='P_o')
            self.P = T.concatenate([self.P_i, self.P_f, self.P_c, self.P_o], axis=1)
            self.non_seq.append(self.P)
            if True:
                self.params.extend([self.P_i, self.P_f, self.P_o])  # P_c is not a param
            else:
                self.params.extend([self.P])

    @property
    def output_shape(self):
        if self.last_only:
            out_shape = (self.input_shape[0], self.n_units)
        else:
            out_shape = (self.input_shape[0], self.input_shape[1], self.n_units)
        return out_shape

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)

        if X.ndim > 3:
            X = T.flatten(X, 3)
        # (n_batch, n_time_steps, n_dim) ->  (n_time_steps, n_batch, n_dim)
        X = X.dimshuffle(1, 0, 2)
        n_batch = X.shape[1]
        # Input dot product is outside of the scan
        X = T.dot(X, self.W) + self.b

        c0 = T.ones((n_batch, self.n_hidden), dtype=floatX)
        h0 = self.activation(c0)

        def one_step(x_t, h_tm1, c_tm1, *args):
            # pre-activation
            if self.peepholes:
                pre_act = x_t + T.dot(h_tm1, self.U) + T.dot(c_tm1, self.P)
            else:
                pre_act = x_t + T.dot(h_tm1, self.U)
            # Clip gradients
            if self.grad_clipping:
                pre_act = theano.gradient.grad_clip(pre_act, -self.grad_clipping, self.grad_clipping)
            # gates
            i_t = sigmoid(pre_act[:, 0: self.n_units])
            f_t = sigmoid(pre_act[:, self.n_units: 2*self.n_units])
            c_t = self.activation(pre_act[:, 2*self.n_units: 3*self.n_units])
            o_t = sigmoid(pre_act[:, 3*self.n_units: 4*self.n_units])

            if self.tied:
                i_t = 1. - f_t
            # cell state
            c_t = f_t * c_tm1 + i_t * c_t
            h_t = o_t * self.activation(c_t)

            return [h_t, c_t]

        [h_vals, _], _ = theano.scan(fn=one_step,
                                     sequences=X,
                                     outputs_info=[h0, c0],
                                     non_sequences=self.non_seq,
                                     go_backwards=self.go_backwards,
                                     allow_gc=self.allow_gc,
                                     strict=True)
        if self.last_only:
            h_vals = h_vals[-1]
        else:
            h_vals = h_vals.dimshuffle(1, 0, 2)
            if self.go_backwards:
                h_vals = h_vals[:, ::-1]

        return h_vals

    def to_conf(self):
        conf = super(LSTM, self).to_conf()
        return conf


class GRU(Layer):
    r"""
    Gated Recurrent unit

    .. math ::
        z_t &= \sigma(x_t.W_z + h_{t-1}.U_z + b_z)\\
        r_t &= \sigma(x_t.W_r + h_{t-1}.U_r + b_r)\\
        \tilde{h_t} &= \tanh(x_t.W_h + (r_t*h_{t-1}).U_h + b_h)\\
        h_t &= (1 - z_t) * h_{t-1} + z_t * \tilde{h_t}

    References
    ----------
    .. [1] http://deeplearning.net/tutorial/lstm.html
    .. [2] https://arxiv.org/pdf/1412.3555.pdf
    .. [3] http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
    """
    n_instances = 0

    def __init__(self, incoming, n_units, activation=tanh, last_only=True, grad_clipping=0,
                 go_backwards=False, allow_gc=False, **kwargs):
        super(GRU, self).__init__(incoming, **kwargs)
        self.allow_gc = allow_gc
        self.grad_clipping = grad_clipping
        self.go_backwards = go_backwards
        self.last_only = last_only
        self.activation = get_activation(activation)
        self.n_feature = self.input_shape[2]  # (n_batch, n_time_steps, n_dim)
        self.n_units = self.n_hidden = self.n_ig = self.n_fg = self.n_cg = self.n_og = n_units
        # update gate
        self.W_z = orthogonal(shape=(self.n_feature, self.n_ig), name='W_z')
        self.U_z = orthogonal(shape=(self.n_hidden, self.n_ig), name='U_z')
        self.b_z = uniform(shape=(self.n_ig,), scale=(-0.5, .5), name='b_z')
        # reset gate
        self.W_r = orthogonal(shape=(self.n_feature, self.n_fg), name='W_r')
        self.U_r = orthogonal(shape=(self.n_hidden, self.n_fg), name='U_r')
        self.b_r = uniform(shape=(self.n_fg,), scale=(0, 1.), name='b_r')
        # output gate
        self.W_o = orthogonal(shape=(self.n_feature, self.n_og), name='W_o')
        self.U_o = orthogonal(shape=(self.n_hidden, self.n_og), name='U_o')
        self.b_o = uniform(shape=(self.n_ig,), scale=(-0.5, .5), name='b_o')
        # Row representation
        self.W = T.concatenate([self.W_z, self.W_r, self.W_o], axis=1)
        self.U = T.concatenate([self.U_z, self.U_r, self.U_o], axis=1)
        self.b = T.concatenate([self.b_z, self.b_r, self.b_o], axis=0)
        # Non sequence for the scan operator
        self.non_seq = [self.U]

        if True:
            self.params.extend([self.W_z, self.U_z, self.b_z,
                                self.W_r, self.U_r, self.b_r,
                                self.W_o, self.U_o, self.b_o])
        else:
            self.params.extend([self.W, self.U, self.b])

    @property
    def output_shape(self):
        if self.last_only:
            out_shape = (self.input_shape[0], self.n_units)
        else:
            out_shape = (self.input_shape[0], self.input_shape[1], self.n_units)
        return out_shape

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)

        if X.ndim > 3:
            X = T.flatten(X, 3)
        # (n_batch, n_time_steps, n_dim) ->  (n_time_steps, n_batch, n_dim)
        X = X.dimshuffle(1, 0, 2)
        n_batch = X.shape[1]
        # Input dot product is outside of the scan
        X = T.dot(X, self.W) + self.b

        c0 = T.ones((n_batch, self.n_hidden), dtype=floatX)
        h0 = self.activation(c0)

        def one_step(x_t, h_tm1, *args):
            # pre-activation
            pre_act = T.dot(h_tm1, self.U)
            # Clip gradients
            if self.grad_clipping:
                pre_act = theano.gradient.grad_clip(pre_act, -self.grad_clipping, self.grad_clipping)
            # gates
            z_t = sigmoid(x_t[:, 0: self.n_units] + pre_act[:, 0: self.n_units])
            r_t = sigmoid(x_t[:, self.n_units: 2*self.n_units] + pre_act[:, self.n_units: 2*self.n_units])
            h_t = x_t[:, 2*self.n_units: 3*self.n_units] + r_t * pre_act[:, 2*self.n_units: 3*self.n_units]

            # hidden state
            h_t = (1 - z_t) * h_tm1 + z_t * h_t

            return h_t

        h_vals, _ = theano.scan(fn=one_step,
                                sequences=X,
                                outputs_info=[h0],
                                non_sequences=self.non_seq,
                                go_backwards=self.go_backwards,
                                allow_gc=self.allow_gc,
                                strict=True)
        if self.last_only:
            h_vals = h_vals[-1]
        else:
            h_vals = h_vals.dimshuffle(1, 0, 2)
            if self.go_backwards:
                h_vals = h_vals[:, ::-1]

        return h_vals

    def to_conf(self):
        conf = super(GRU, self).to_conf()
        return conf


class BNLSTM(LSTM):
    r"""
    Batch Normalization Long Short Term Memory

    .. math ::
        i_t &= \sigma(x_t.W_i + h_{t-1}.U_i + b_i)\\
        f_t &= \sigma(x_t.W_f + h_{t-1}.U_f + b_f)\\
        \tilde{C_t} &= \tanh(x_t.W_c + h_{t-1}.U_c + b_c)\\
        C_t &= f_t * C_{t-1} + i_t * \tilde{C_t}\\
        o_t &= \sigma(x_t.W_o + h_{t-1}.U_o + b_o)\\
        h_t &= o_t * \tanh(C_t) && \text{Hidden state}\\

    Parameters
    ----------
    incoming : a `Layer`
        The incoming layer with an output_shape = (n_batches, n_time_steps, n_dim)
    n_units : int
        n_hidden = n_input_gate = n_forget_gate = n_cell_gate = n_output_gate = n_units
        All gates have the same number of units
    n_out : int
        number of output units
    activation : `yadll.activations` function default is `yadll.activations.tanh`
        activation function
    last_only : boolean default is True
        set to true if you only need the last element of the output sequence.
        Theano will optimize graph.

    References
    ----------
    .. [1] https://arxiv.org/pdf/1603.09025.pdf
    """
    n_instances = 0

    def __init__(self, incoming, n_units, activation=tanh, last_only=True, grad_clipping=0,
                 go_backwards=False, allow_gc=False, **kwargs):
        super(BNLSTM, self).__init__(incoming, n_units, activation=activation, last_only=last_only,
                                     grad_clipping=grad_clipping, go_backwards=go_backwards,
                                     allow_gc=allow_gc, **kwargs)
        # Batch Normalise the input
        self.bn_x = BatchNormalization(None, nested=True)
        self.bn_x.init_params(input_shape=(self.input_shape[1], self.input_shape[0], n_units), beta=False)
        self.params.extend(self.bn_x.params)
        # Batch Normalise the hidden state
        self.bn_h = BatchNormalization(None, nested=True)
        self.bn_h.init_params(input_shape=(self.input_shape[1], self.input_shape[0], n_units), beta=False)
        self.params.extend(self.bn_h.params)
        # Batch Normalise the cell state
        self.bn_c = BatchNormalization(None, nested=True)
        self.bn_c.init_params(input_shape=(self.input_shape[1], self.input_shape[0], n_units), beta=False)
        self.params.extend(self.bn_c.params)

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)

        if X.ndim > 3:
            X = T.flatten(X, 3)
        # (n_batch, n_time_steps, n_dim) ->  (n_time_steps, n_batch, n_dim)
        X = X.dimshuffle(1, 0, 2)
        n_batch = X.shape[1]
        # Input dot product is outside of the scan
        X = T.dot(X, self.W)
        # Batch Normalise the input
        self.bn_x.input_layer = X
        X = self.bn_x.get_output(**kwargs) + self.b

        c0 = T.ones((n_batch, self.n_hidden), dtype=floatX)
        h0 = self.activation(c0)

        def one_step(x_t, h_tm1, c_tm1, *args):
            H = T.dot(h_tm1, self.U)
            # Batch Normalise the hidden state
            self.bn_h.input_layer = H
            H = self.bn_h.get_output(**kwargs)
            # pre-activation
            if self.peepholes:
                pre_act = x_t + H + T.dot(c_tm1, self.P)
            else:
                pre_act = x_t + H
            # Clip gradients
            if self.grad_clipping:
                pre_act = theano.gradient.grad_clip(pre_act, -self.grad_clipping, self.grad_clipping)
            # gates
            i_t = sigmoid(pre_act[:, 0: self.n_units])
            f_t = sigmoid(pre_act[:, self.n_units: 2*self.n_units])
            c_t = self.activation(pre_act[:, 2*self.n_units: 3*self.n_units])
            o_t = sigmoid(pre_act[:, 3*self.n_units: 4*self.n_units])

            if self.tied:
                i_t = 1. - f_t
            # cell state
            c_t = f_t * c_tm1 + i_t * c_t
            # Batch Normalise the cell state
            self.bn_c.input_layer = c_t
            c_t = self.bn_c.get_output(**kwargs)
            h_t = o_t * self.activation(c_t)

            return [h_t, c_t]

        [h_vals, _], _ = theano.scan(fn=one_step,
                                     sequences=X,
                                     outputs_info=[h0, c0],
                                     non_sequences=self.non_seq,
                                     go_backwards=self.go_backwards,
                                     allow_gc=self.allow_gc,
                                     strict=True)
        if self.last_only:
            h_vals = h_vals[-1]
        else:
            h_vals = h_vals.dimshuffle(1, 0, 2)
            if self.go_backwards:
                h_vals = h_vals[:, ::-1]

        return h_vals

    def to_conf(self):
        conf = super(BNLSTM, self).to_conf()
        return conf
