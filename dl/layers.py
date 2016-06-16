# -*- coding: UTF-8 -*-
"""
All the neural network layers currently supported by dl.
"""
from .init import *
from .objectives import *
from .updates import *
from .utils import *

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

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
        The layer name.


    """
    def __init__(self, incoming, name=None):
        """
        The base class that represent a single layer of any neural network.
        It has to be subclassed by any kind of layer.


        """

        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        self.params = []
        self.reguls = 0

    def get_params(self):
        """
        Return a list of Theano shared variables representing the parameters of
        this layer.

        :return: list of Theano shared variables that parametrize the layer
        """
        return self.params

    def get_reguls(self):
        """
        Return Theano expression representing the sum of the regulators of
        this layer.

        Returns
         Theano expression representing the sum of the regulators
         of this layer
        """
        return self.reguls

    @property
    def output_shape(self):
        """
        Compute the output shape of this layer given the input shape.

        :return: a tuple representing the shape of the output of this layer.

        ..note:: This method has to be overriden by new layer implementation or
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


class InputLayer(Layer):
    def __init__(self, shape, input_var=None, **kwargs):
        """
        The input layer of any network

        Parameters
        ----------
        shape : `tuple` of `int`
            The shape of the input layer
        input_var : `Theano shared Variables`, optional
            The input data of the network
        """
        super(InputLayer, self).__init__(shape, **kwargs)
        self.input = input_var

    def get_output(self, **kwargs):
        return self.input


class ReshapeLayer(Layer):
    def __init__(self, incoming, output_shape=None, **kwargs):
        super(ReshapeLayer, self).__init__(incoming, **kwargs)
        self.reshape_shape = output_shape

    @property
    def output_shape(self):
        return self.reshape_shape

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return X.reshape(self.reshape_shape)


class FlattenLayer(Layer):
    def __init__(self, incoming, ndim=2, **kwargs):
        super(FlattenLayer, self).__init__(incoming, **kwargs)
        self.ndim = ndim

    @property
    def output_shape(self):
        return self.input_shape[0], np.prod(self.input_shape[1:])

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return X.flatten(self.ndim)


class DenseLayer(Layer):
    def __init__(self, incoming, nb_units, W=glorot_uniform, b=constant,
                 activation=tanh, l1=None, l2=None, **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
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
        self.activation = activation
        if l1:
            self.reguls += l1 * T.mean(T.abs_(self.W))
        if l2:
            self.reguls += l2 * T.mean(T.sqr(self.W))

    @property
    def output_shape(self):
        return self.input_shape[0], self.shape[1]

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return self.activation(T.dot(X, self.W) + self.b)


class UnsupervisedLayer(DenseLayer):
    def __init__(self, incoming, nb_units, hyperparameters, **kwargs):
        super(UnsupervisedLayer, self).__init__(incoming, nb_units, **kwargs)
        self.hp = hyperparameters
        self.unsupervised_params = list(self.params)

    def get_encoded_input(self, **kwargs):
        raise NotImplementedError

    def get_unsupervised_cost(self, **kwargs):
        raise NotImplementedError

    @timer(' Pretraining the layer')
    def unsupervised_training(self, x, train_set_x):
        logger.info('... Pretraining the layer: %s' % self.name)
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
            logger.info('Layer: %s, pretraining epoch %d, cost %d' % (self.name, epoch, np.mean(c)))


class LogisticRegression(DenseLayer):
    def __init__(self, incoming, nb_class, W=constant, activation=softmax, **kwargs):
        super(LogisticRegression, self).__init__(incoming, nb_class, W=W,
                                                 activation=activation, **kwargs)


class Dropout(Layer):
    def __init__(self, incoming, corruption_level=0.5, **kwargs):
        super(Dropout, self).__init__(incoming, **kwargs)
        self.p = 1 - corruption_level

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.p > 0 and stochastic:
            X = X * T_rng.binomial(self.input_shape, n=1, p=self.p, dtype=floatX)
        return X


class Dropconnect(DenseLayer):
    def __init__(self, incoming, nb_units, corruption_level=0.5, **kwargs):
        super(Dropconnect, self).__init__(incoming, nb_units, **kwargs)
        self.p = 1 - corruption_level

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.p > 0 and stochastic:
            self.W = self.W * T_rng.binomial(self.shape, n=1, p=self.p, dtype=floatX)
        return self.activation(T.dot(X, self.W) + self.b)


class PoolLayer(Layer):
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
    def __init__(self, incoming, poolsize, image_shape=None, filter_shape=None,
                  b=constant, activation=tanh, **kwargs):
        super(ConvPoolLayer, self).__init__(incoming, poolsize=poolsize, image_shape=image_shape,
                                            filter_shape=filter_shape, pool_scale=poolsize, **kwargs)
        self.b = initializer(b, shape=(self.filter_shape[0],), name='b')
        self.params.append(self.b)
        self.activation = activation

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
                X = X + T_rng.normal(self.input_shape, avg=0.0, std=self.sigma, dtype=floatX)
            else:
                X = X * T_rng.binomial(self.input_shape, n=1, p=self.p, dtype=floatX)
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


class RNN(Layer):
    def __init__(self, incoming, n_hidden, n_out, activation=tanh, **kwargs):
        super(RNN, self).__init__(incoming, **kwargs)
        self.activation = activation
        if isinstance(n_hidden, tuple):
            self.n_hidden, self.n_i, self.n_c, self.n_o, self.n_f = n_hidden
        else:
            self.n_hidden = self.n_i = self.n_c = self.n_o = self.n_f = n_hidden
        self.n_in = self.input_shape[1]
        self.n_out = n_out

        self.W_x = orthogonal(shape=(self.n_in, self.n_f), name='W_x')
        self.W_h = orthogonal(shape=(n_hidden, self.n_f), name='W_h')
        self.b_h = uniform(shape=self.n_f, scale=(0, 1.), name='b_h')
        self.W_y = orthogonal(shape=(n_hidden, n_out), name='W_y')
        self.b_y = constant(shape=self.n_out, name='b_y')

        self.params.extend([self.W_x, self.W_h, self.b_h,
                            self.W_y, self.b_y])

        self.c0 = constant(shape=self.n_hidden, name='c0')
        self.h0 = activation(self.c0)

    def one_step(self, x_t, h_tm1, W_x, W_h, b_h, W_y, b_y):
        ha_t = T.dot(x_t, W_x) + T.dot(h_tm1, W_h) + b_h
        h_t = self.activation(ha_t)
        s_t = T.dot(h_t, W_y) + b_y

        return [h_t, s_t]

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        [h_t, s_t], updates = theano.scan(fn=self.one_step,
                                          sequences=X,
                                          outputs_info=[self.h0, None],
                                          non_sequences=self.params,
                                          allow_gc=False,
                                          strict=True)
        return s_t


class LSTM(Layer):
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


        self.c0 = constant(shape=self.n_hidden, name='c0')
        self.h0 = activation(self.c0)

    def one_step(self, x_t, h_tm1, c_tm1, W_xi, W_hi, b_i,
                                          W_xf, W_hf, b_f,
                                          W_xc, W_hc, b_c,
                                          W_xo, W_ho, b_o,
                                          W_hy, b_y):
        # forget gate
        f_t = sigmoid(T.dot(x_t, W_xf) + T.dot(h_tm1, W_hf) + b_f)
        # input gate
        i_t = sigmoid(T.dot(x_t, W_xi) + T.dot(h_tm1, W_hi) + b_i)

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
                                             sequences=X,
                                             outputs_info=[self.h0, self.c0, None],
                                             non_sequences=self.params)
        return y_vals

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
