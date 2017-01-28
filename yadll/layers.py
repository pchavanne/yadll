# -*- coding: UTF-8 -*-
"""
All the neural network layers currently supported by yaddll.
"""
from .init import *
from .objectives import *
from .updates import *
from .utils import *

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
import yadll
import logging

logger = logging.getLogger(__name__)

T_rng = RandomStreams(np_rng.randint(2 ** 30))


class Layer(object):
    """
    Layer is the base class of any neural network layer.
    It has to be subclassed by any kind of layer.

    Parameters
    ----------
    incoming : a `Layer` or a `tuple` of `int`
        The incoming layer or shape if input layer
    name : `string`, optional
        The layer name. default name is the class name
        plus instantiation number i.e: 'DenseLayer 3'

    """
    nb_instances = 0

    def __init__(self, incoming, name=None, **kwargs):
        """
        The base class that represent a single layer of any neural network.
        It has to be subclassed by any kind of layer.

        """
        self.id = self.get_id()
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        if name is None:
            self.name = self.__class__.__name__ + ' ' + str(self.id)
        self.params = []
        self.reguls = 0

    @classmethod
    def get_id(cls):
        cls.nb_instances += 1
        return cls.nb_instances

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
        conf = self.__dict__.copy()
        for key in ['id', 'params', 'reguls']:
            conf.pop(key, None)
        if conf['input_layer']:
            conf['input_layer'] = conf['input_layer'].name
        if 'activation' in conf:
            conf['activation'] = conf['activation'].__name__
        conf['type'] = self.__class__.__name__
        return conf

    def from_conf(self, conf):
        self.__dict__.update(conf)


class InputLayer(Layer):
    """
    Input layer of the data, it has no parameters, it just shapes the data as
    the input for any network. A ::class:`InputLayer` is always the first layer of any network.
    """
    nb_instances = 0

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


class ReshapeLayer(Layer):
    """
    Reshape the incoming layer to the output_shape.
    """
    nb_instances = 0

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


class FlattenLayer(Layer):
    """
    Reshape layers back to flat
    """
    nb_instances = 0

    def __init__(self, incoming, ndim=2, **kwargs):
        super(FlattenLayer, self).__init__(incoming, **kwargs)
        self.ndim = ndim

    @property
    def output_shape(self):
        return self.input_shape[0], np.prod(self.input_shape[1:])

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return X.flatten(self.ndim)


class Activation(Layer):
    """
    Apply activation function to previous layer
    """
    nb_instances = 0

    def __init__(self, incoming, activation=linear, **kwargs):
        super(Activation, self).__init__(incoming, **kwargs)
        self.activation = get_activation(activation)

    @property
    def output_shape(self):
        return self.input_shape[0], np.prod(self.input_shape[1:])

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return self.activation(X)


class DenseLayer(Layer):
    """
    Fully connected neural network layer
    """
    nb_instances = 0

    def __init__(self, incoming, nb_units, W=glorot_uniform, b=constant,
                 activation=tanh, l1=None, l2=None, **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nb_units = nb_units
        self.shape = (self.input_shape[1], nb_units)
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


class UnsupervisedLayer(DenseLayer):
    """
    Base class for all unsupervised layers.
    Unsupervised layers are pre-trained against its own input.

    """
    nb_instances = 0

    def __init__(self, incoming, nb_units, hyperparameters, **kwargs):
        super(UnsupervisedLayer, self).__init__(incoming, nb_units, **kwargs)
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


class LogisticRegression(DenseLayer):
    """
    Dense layer with softmax activation

    References
    ----------
    .. [1] http://deeplearning.net/tutorial/logreg.html
    """
    nb_instances = 0

    def __init__(self, incoming, nb_class, W=constant, activation=softmax, **kwargs):
        super(LogisticRegression, self).__init__(incoming, nb_class, W=W,
                                                 activation=activation, **kwargs)

    def to_conf(self):
        conf = super(LogisticRegression, self).to_conf()
        conf['nb_class'] = conf.pop('nb_units')
        return conf


class Dropout(Layer):
    """
    Dropout layer
    """
    nb_instances = 0

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


class Dropconnect(DenseLayer):
    """
    DropConnect layer
    """
    nb_instances = 0

    def __init__(self, incoming, nb_units, corruption_level=0.5, **kwargs):
        super(Dropconnect, self).__init__(incoming, nb_units, **kwargs)
        self.p = 1 - corruption_level

    def get_output(self, stochastic=True, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.p != 1 and stochastic:
            self.W = self.W * T_rng.binomial(self.shape, n=1, p=self.p, dtype=floatX)
        return self.activation(T.dot(X, self.W) + self.b)


class PoolLayer(Layer):
    """
    Pooling layer, default is maxpooling
    """
    nb_instances = 0

    def __init__(self, incoming, poolsize, stride=None, ignore_border=True,
                 padding=(0, 0), mode='max', **kwargs):
        super(PoolLayer, self).__init__(incoming, **kwargs)
        self.poolsize = poolsize
        self.stride = stride    # If st is None, it is considered equal to ds
        self.ignore_border = ignore_border
        self.padding = padding
        self.mode = mode    # {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}

    def pool(self, input, ds):
        return pool.pool_2d(input=input, ds=ds, st=self.stride, ignore_border=self.ignore_border,
                                      padding=self.padding, mode=self.mode)

    @property
    def output_shape(self):
        return (self.input_shape[0],
                self.input_shape[1],
                self.input_shape[2] / self.poolsize[0],
                self.input_shape[3] / self.poolsize[1])

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        return self.pool(input=X, ds=self.poolsize)


class ConvLayer(Layer):
    """
    Convolutional layer
    """
    nb_instances = 0

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
        if pool_scale:
            self.fan_out = self.fan_out / np.prod(pool_scale)
        self.W = initializer(W, shape=self.filter_shape, fan=(self.fan_in, self.fan_out), name='W')
        self.params.append(self.W)
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


class ConvPoolLayer(ConvLayer, PoolLayer):
    """
    Convolutional and pooling layer

    References
    ----------
    .. [1] http://deeplearning.net/tutorial/lenet.html
    """
    nb_instances = 0

    def __init__(self, incoming, poolsize, image_shape=None, filter_shape=None,
                  b=constant, activation=tanh, **kwargs):
        super(ConvPoolLayer, self).__init__(incoming, poolsize=poolsize, image_shape=image_shape,
                                            filter_shape=filter_shape, pool_scale=poolsize, **kwargs)
        self.b = initializer(b, shape=(self.filter_shape[0],), name='b')
        self.params.append(self.b)
        self.activation = get_activation(activation)

    @property
    def output_shape(self):
        return (self.input_shape[0],
                self.filter_shape[0],
                (self.image_shape[2] - self.filter_shape[2] + 1) / self.poolsize[0],
                (self.image_shape[3] - self.filter_shape[3] + 1) / self.poolsize[1])

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        conv_X = self.conv(input=X, filters=self.W, image_shape=self.image_shape,
                           filter_shape=self.filter_shape)
        pool_X = self.pool(input=conv_X, ds=self.poolsize)
        return self.activation(pool_X + self.b.dimshuffle('x', 0, 'x', 'x'))


class AutoEncoder(UnsupervisedLayer):
    """
    Autoencoder

    References
    ----------
    .. [1] http://deeplearning.net/tutorial/dA.html
    """
    nb_instances = 0

    def __init__(self, incoming, nb_units, hyperparameters, corruption_level=0.0,
                 W=(glorot_uniform, {'gain': sigmoid}), b_prime=constant,
                 sigma=None, contraction_level= None, **kwargs):
        super(AutoEncoder, self).__init__(incoming, nb_units, hyperparameters, W=W, **kwargs)
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
        cost = T.mean(categorical_crossentropy(Z, X))
        if self.contraction_level:
            # For sigmoid: J = Y * (1 - Y) * W
            # For tanh: J = (1 + Y) * (1 - Y) * W
            J = T.reshape(Y * (1 - Y), (self.hp.batch_size, 1, self.shape[1])) \
                * T.reshape(self.W, (1, self.shape[0], self.shape[1]))
            Lj = T.sum(J**2) / self.hp.batch_size
            cost = cost + self.contraction_level * T.mean(Lj)
        return cost


class RBM(UnsupervisedLayer):
    """
    Restricted Boltzmann Machines

    References
    ----------
    .. [1] http://deeplearning.net/tutorial/rbm.html
    """
    nb_instances = 0

    def __init__(self, incoming, nb_units, hyperparameters, W=glorot_uniform,
                 b_hidden=constant, activation=sigmoid, **kwargs):
        super(RBM, self).__init__(incoming, nb_units, hyperparameters, W=W,
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


class BatchNormalization(Layer):
    r"""
    Normalize the input layer over each mini-batch according to [1]_:

    .. math::
        \hat{x} = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}}

        y = \gamma * \hat{x} + \beta

    References
    ----------

    .. [1] http://jmlr.org/proceedings/papers/v37/ioffe15.pdf
    """
    nb_instances = 0

    def __init__(self, incoming, axis=-2, alpha=0.1, epsilon=1e-5, **kwargs):
        super(BatchNormalization, self).__init__(incoming, **kwargs)
        self.axis = axis
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = initializer(constant, shape=self.input_shape, value=1, name='gamma')
        self.params.append(self.gamma)
        self.beta = initializer(constant, shape=(self.input_shape[1],), value=0, name='beta')
        self.params.append(self.beta)
        self.mean = initializer(constant, shape=self.input_shape, value=0, name='mean')
        self.var = initializer(constant, shape=self.input_shape, value=1, name='var')

    def get_output(self, stochastic=True, **kwargs):
        x = self.input_layer.get_output(**kwargs)
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


class RNN(Layer):
    r"""
    Recurrent Neural Network

    .. math ::
        h_t = \sigma(x_t.W + h_{t-1}.U + b)

    References
    ----------

    .. [1] http://deeplearning.net/tutorial/rnnslu.html
    """
    nb_instances = 0

    def __init__(self, incoming, n_hidden, n_out, activation=sigmoid, last_only=True, **kwargs):
        super(RNN, self).__init__(incoming, **kwargs)
        self.last_only = last_only
        self.activation = get_activation(activation)

        self.n_in = self.input_shape[1]
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.W = orthogonal(shape=(self.n_in, self.n_out), name='W')
        self.U = orthogonal(shape=(self.n_hidden, self.n_out), name='U')
        self.b = uniform(shape=self.n_out, scale=(0, 1.), name='b')

        self.params.extend([self.W, self.U, self.b])

        self.c0 = constant(shape=self.n_hidden, name='c0')
        self.h0 = self.activation(self.c0)

    def one_step(self, x_t, h_tm1, W, U, b):
        h_t = self.activation(T.dot(x_t, W) + T.dot(h_tm1, U) + b)
        return h_t

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        h_t, updates = theano.scan(fn=self.one_step,
                                          sequences=X,
                                          outputs_info=[self.h0, None],
                                          non_sequences=self.params,
                                          allow_gc=False,
                                          strict=True)
        return h_t


class LSTM(Layer):
    r"""
    Long Short Term Memory

    .. math ::
        i_t &= \sigma(x_t.W_i + h_{t-1}.U_i + b_i) && \text{Input gate}\\
        f_t &= \sigma(x_t.W_f + h_{t-1}.U_f + b_f) && \text{Forget gate}\\
        \tilde{C_t} &= \tanh(x_t.W_c + h_{t-1}.U_c + b_c) && \text{Cell gate}\\
        C_t &= f_t * C_{t-1} + i_t * \tilde{C_t} && \text{Cell state}\\
        o_t &= \sigma(x_t.W_o + h_{t-1}.U_o + b_o) && \text{Output gate}\\
        h_t &= o_t * \tanh(C_t) && \text{Hidden state}\\

    with Peephole connections:

    .. math ::
        i_t &= \sigma(x_t.W_i + h_{t-1}.U_i + C_{t-1}.P_i + b_i) && \text{Input gate}\\
        f_t &= \sigma(x_t.W_f + h_{t-1}.U_f + C_{t-1}.P_f + b_f) && \text{Forget gate}\\
        \tilde{C_t} &= \tanh(x_t.W_c + h_{t-1}.U_c + b_c) && \text{Cell gate}\\
        C_t &= f_t * C_{t-1} + i_t * \tilde{C_t} & \text{Cell state}\\
        o_t &= \sigma(x_t.W_o + h_{t-1}.U_o + C_t.P_o + b_o) && \text{Output gate}\\
        h_t &= o_t * \tanh(C_t) && \text{Hidden state}\\

    with tied forget and input gates:

    .. math ::
        C_t &= f_t * C_{t-1} + (1 - f_t) * \tilde{C_t} && \text{Cell state}\\

    Parameters
    ----------
    incoming : a `Layer`
        The incoming layer
    n_hidden : int or tuple of int
        (n_hidden, n_input, n_forget, n_cell, n_output).
        If an int is provided all gates have the same number of units
    n_out : int
        number of output units
    peephole : boolean default is False
        use peephole connections.
    tied_i : boolean default is false
        tie input and forget gate
    activation : `yadll.activations` function default is `yadll.activations.tanh`
        activation function
    last_only : boolean default is True
        set to true if you only need the last element of the output sequence

    References
    ----------
    .. [1] http://deeplearning.net/tutorial/lstm.html
    .. [2] http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php
    .. [3] http://people.idsia.ch/~juergen/lstm/
    .. [4] http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    .. [5] https://arxiv.org/pdf/1308.0850v5.pdf
    """
    nb_instances = 0

    def __init__(self, incoming, n_hidden, n_out, peephole=False, tied_i_f=False, activation=tanh, last_only=True, **kwargs):
        super(LSTM, self).__init__(incoming, **kwargs)
        self.last_only = last_only
        self.peephole = peephole    # input and forget gates layers look at the cell state
        self.tied = tied_i_f        # only input new values to the state when we forget something
        self.activation = get_activation(activation)
        if isinstance(n_hidden, tuple):
            self.n_hidden, self.n_i, self.n_f, self.n_c, self.n_o = n_hidden
        else:
            self.n_hidden = self.n_i = self.n_f = self.n_c = self.n_o = n_hidden
        self.n_in = self.input_shape[1]
        self.n_out = n_out
        # input gate
        self.W_i = orthogonal(shape=(self.n_in, self.n_i), name='W_i')
        self.U_i = orthogonal(shape=(self.n_hidden, self.n_i), name='U_i')
        self.b_i = uniform(shape=self.n_i, scale=(-0.5, .5), name='b_i')
        # forget gate
        self.W_f = orthogonal(shape=(self.n_in, self.n_f), name='W_f')
        self.U_f = orthogonal(shape=(self.n_hidden, self.n_f), name='U_f')
        self.b_f = uniform(shape=self.n_f, scale=(0, 1.), name='b_f')
        # cell state
        self.W_c = orthogonal(shape=(self.n_in, self.n_c), name='W_c')
        self.U_c = orthogonal(shape=(self.n_hidden, self.n_c), name='U_c')
        self.b_c = constant(shape=self.n_c, name='b_c')
        # output gate
        self.W_o = orthogonal(shape=(self.n_in, self.n_o), name='W_o')
        self.U_o = orthogonal(shape=(self.n_hidden, self.n_o), name='U_o')
        self.b_o = uniform(shape=self.n_i, scale=(-0.5, .5), name='b_o')

        self.params.extend([self.W_i, self.U_i, self.b_i,
                            self.W_f, self.U_f, self.b_f,
                            self.W_c, self.U_c, self.b_c,
                            self.W_o, self.U_o, self.b_o])

        self.W = T.concatenate([self.W_i, self.W_f, self.W_c, self.W_o])
        self.U = T.concatenate([self.U_i, self.U_f, self.U_c, self.U_o])
        self.b = T.concatenate([self.b_i, self.b_f, self.b_c, self.b_o])

        self.c0 = constant(shape=self.n_hidden, name='c0')
        self.h0 = self.activation(self.c0)

    def one_step(self, x_t, h_tm1, c_tm1, W_i, U_i, b_i,
                                          W_f, U_f, b_f,
                                          W_c, U_c, b_c,
                                          W_o, U_o, b_o):
        # forget gate
        f_t = sigmoid(T.dot(x_t, W_f) + T.dot(h_tm1, U_f) + b_f)
        # input gate
        i_t = sigmoid(T.dot(x_t, W_i) + T.dot(h_tm1, U_i) + b_i)

        # cell state
        c_tt = self.activation(T.dot(x_t, W_c) + T.dot(h_tm1, U_c) + b_c)
        c_t = f_t * c_tm1 + i_t * c_tt
        if self.tied:
            c_t = f_t * c_tm1 + (1 - f_t) * c_tt

        # output gate
        o_t = sigmoid(T.dot(x_t, W_o) + T.dot(h_tm1, U_o) + b_o)
        # if self.peephole:
        #     o_t = sigmoid(T.dot(x_t, W_o) + T.dot(h_tm1, U_o) + T.dot(c_t, W_co) + b_o)

        h_t = o_t * self.activation(c_t)

        return [h_t, c_t]

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        [h_vals, _], _ = theano.scan(fn=self.one_step,
                                             sequences=X,
                                             outputs_info=[self.h0, self.c0, None],
                                             non_sequences=self.params,
                                             allow_gc=False,
                                             strict=True)
        return h_vals


class LSTM_Old(Layer):
    def __init__(self, incoming, n_hidden, n_out, peephole=False, tied_i_f=False, activation=tanh, **kwargs):
        super(LSTM, self).__init__(incoming, **kwargs)
        self.peephole = peephole    # gate layers look at the cell state
        self.tied = tied_i_f        # only input new values to the state when we forget something
        self.activation = activation
        if isinstance(n_hidden, tuple):
            self.n_hidden, self.n_i, self.n_c, self.n_o, self.n_f = n_hidden
        else:
            self.n_hidden = self.n_i = self.n_c = self.n_o = self.n_f = n_hidden
        self.n_in = self.input_shape[1]
        self.n_out = n_out
        # forget gate
        self.W_xf = orthogonal(shape=(self.n_in, self.n_f), name='W_xf')
        self.W_hf = orthogonal(shape=(n_hidden, self.n_f), name='W_hf')
        self.b_f = uniform(shape=self.n_f, scale=(0, 1.), name='b_f')
        # input gate
        self.W_xi = orthogonal(shape=(self.n_in, self.n_i), name='W_xi')
        self.W_hi = orthogonal(shape=(self.n_hidden, self.n_i), name='W_hi')
        self.b_i = uniform(shape=self.n_i, scale=(-0.5,.5), name='b_i')
        # cell state
        self.W_xc = orthogonal(shape=(self.n_in, self.n_c), name='W_xc')
        self.W_hc = orthogonal(shape=(n_hidden, self.n_c), name='W_hc')
        self.b_c = constant(shape=self.n_c, name='b_c')
        # output gate
        self.W_xo = orthogonal(shape=(self.n_in, self.n_o), name='W_x0')
        self.W_ho = orthogonal(shape=(self.n_hidden, self.n_o), name='W_ho')
        self.b_o = uniform(shape=self.n_i, scale=(-0.5,.5), name='b_o')

        self.W_hy = orthogonal(shape=(n_hidden, n_out), name='W_hy')
        self.b_y = constant(shape=self.n_out, name='b_y')

        self.params.extend([self.W_xi, self.W_hi, self.b_i,
                            self.W_xf, self.W_hf, self.b_f,
                            self.W_xc, self.W_hc, self.b_c,
                            self.W_xo, self.W_ho, self.b_o,
                            self.W_hy, self.b_y])
        if peephole:
            self.W_cf = orthogonal(shape=(self.n_c, self.n_f), name='W_cf')
            self.W_ci = orthogonal(shape=(self.n_c, self.n_i), name='W_ci')
            self.W_co = orthogonal(shape=(self.n_c, self.n_o), name='W_co')
            self.params.extend([self.W_cf, self.W_ci, self.W_co])

        self.c0 = constant(shape=self.n_hidden, name='c0')
        self.h0 = activation(self.c0)

    def one_step(self, x_t, h_tm1, c_tm1, W_xi, W_hi, b_i,
                                          W_xf, W_hf, b_f,
                                          W_xc, W_hc, b_c,
                                          W_xo, W_ho, b_o,
                                          W_hy, b_y,
                                          W_cf=None, W_ci=None, W_co=None):
        # forget gate
        f_t = sigmoid(T.dot(x_t, W_xf) + T.dot(h_tm1, W_hf) + b_f)
        # input gate
        i_t = sigmoid(T.dot(x_t, W_xi) + T.dot(h_tm1, W_hi) + b_i)
        if self.peephole:
            f_t = sigmoid(T.dot(x_t, W_xf) + T.dot(h_tm1, W_hf) + T.dot(c_tm1, W_cf) + b_f)
            i_t = sigmoid(T.dot(x_t, W_xi) + T.dot(h_tm1, W_hi) + T.dot(c_tm1, W_ci) + b_i)

        # cell state
        c_tt = self.activation(T.dot(x_t, W_xc) + T.dot(h_tm1, W_hc) + b_c)
        c_t = f_t * c_tm1 + i_t * c_tt
        if self.tied:
            c_t = f_t * c_tm1 + (1 - f_t) * c_tt

        # output gate
        o_t = sigmoid(T.dot(x_t, W_xo) + T.dot(h_tm1, W_ho) + b_o)
        if self.peephole:
            o_t = sigmoid(T.dot(x_t, W_xo) + T.dot(h_tm1, W_ho) + T.dot(c_t, W_co) + b_o)

        h_t = o_t * self.activation(c_t)

        y_t = sigmoid(T.dot(h_t, W_hy) + b_y)

        return [h_t, c_t, y_t]

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        [h_vals, _, y_vals], _ = theano.scan(fn=self.one_step,
                                             sequences=dict(input=X, taps=[0]),
                                             outputs_info=[self.h0, self.c0, None],
                                             non_sequences=self.params,
                                             allow_gc=False,
                                             strict=True)
        return y_vals


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
    """